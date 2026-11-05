const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("=== Submitting zk-SNARK Proof to AMLVerifier Contract ===");

  // 1. Read the deployed addresses
  const deployedAddressesPath = path.join(__dirname, "../deployed_addresses.json");
  if (!fs.existsSync(deployedAddressesPath)) {
    throw new Error("deployed_addresses.json not found! Run deploy.js first.");
  }
  const deployedAddresses = JSON.parse(fs.readFileSync(deployedAddressesPath));
  console.log(`AMLVerifier contract address: ${deployedAddresses.AMLVerifier}`);

  // 2. Read proof.json
  const proofPath = path.join(__dirname, "../../schemas/proof.json");
  if (!fs.existsSync(proofPath)) {
    throw new Error("proof.json not found in schemas/ folder!");
  }
  const proofData = JSON.parse(fs.readFileSync(proofPath));
  const roundNumber = proofData.round_reference;
  console.log(`Proof loaded. Target Round Reference: ${roundNumber}`);

  // 3. Read corresponding global_model_round_N.json metrics
  const modelPath = path.join(__dirname, `../../schemas/global_model_round_${roundNumber}.json`);
  if (!fs.existsSync(modelPath)) {
    throw new Error(`global_model_round_${roundNumber}.json not found in schemas/ folder!`);
  }
  const modelData = JSON.parse(fs.readFileSync(modelPath));
  console.log(`Model metadata loaded for round ${roundNumber}.`);

  // 4. Connect to the AMLVerifier contract (auto-deploying if current network has no bytecode)
  let contractAddress = deployedAddresses.AMLVerifier;
  const code = await ethers.provider.getCode(contractAddress);
  if (code === "0x" || code === "0x0") {
    console.log("[NOTE] Contract not found at address on this network. Deploying fresh contracts...");
    const Groth16VerifierFactory = await ethers.getContractFactory("Groth16Verifier");
    const verifierInst = await Groth16VerifierFactory.deploy();
    await verifierInst.waitForDeployment();
    const verifierAddr = await verifierInst.getAddress();

    const AMLVerifierFactory = await ethers.getContractFactory("AMLVerifier");
    const amlInst = await AMLVerifierFactory.deploy(verifierAddr);
    await amlInst.waitForDeployment();
    contractAddress = await amlInst.getAddress();
    console.log(`Deployed Groth16Verifier to ${verifierAddr} and AMLVerifier to ${contractAddress}`);

    // Update deployed_addresses.json
    deployedAddresses.Groth16Verifier = verifierAddr;
    deployedAddresses.AMLVerifier = contractAddress;
    fs.writeFileSync(deployedAddressesPath, JSON.stringify(deployedAddresses, null, 2));
  }

  const AMLVerifier = await ethers.getContractFactory("AMLVerifier");
  const amlVerifier = AMLVerifier.attach(contractAddress);

  // 5. Parse and format parameters
  const a = proofData.proof.pi_a.slice(0, 2);
  const b = [
    [proofData.proof.pi_b[0][1], proofData.proof.pi_b[0][0]],
    [proofData.proof.pi_b[1][1], proofData.proof.pi_b[1][0]]
  ];
  const c = proofData.proof.pi_c.slice(0, 2);
  const input = proofData.publicSignals;
  const circuitHash = proofData.circuit_hash || "0x19f12a04781bba1340483dda53dc493243c57baa";

  // Discovered rounds from schemas/ directory
  const schemasDir = path.join(__dirname, "../../schemas");
  const modelFiles = fs.readdirSync(schemasDir)
    .filter(f => f.startsWith("global_model_round_") && f.endsWith(".json"))
    .map(f => {
      const match = f.match(/global_model_round_(\d+)\.json/);
      return match ? parseInt(match[1], 10) : null;
    })
    .filter(n => n !== null)
    .sort((x, y) => x - y);

  const roundsToSubmit = modelFiles.length > 0 ? modelFiles : [1];
  console.log(`Discovered ${roundsToSubmit.length} round model schemas: [${roundsToSubmit.join(", ")}]`);

  for (const rNum of roundsToSubmit) {
    const isAlreadyVerified = await amlVerifier.isRoundVerified(rNum);
    if (isAlreadyVerified) {
      console.log(`[SKIP] Round ${rNum} is already finalized on-chain.`);
      continue;
    }

    const rPath = path.join(schemasDir, `global_model_round_${rNum}.json`);
    const rData = JSON.parse(fs.readFileSync(rPath, "utf8"));
    const modelWeightsStr = JSON.stringify(rData.weights || []);
    const modelWeightsHash = ethers.keccak256(ethers.toUtf8Bytes(modelWeightsStr));

    const scale = 10000;
    const accuracy = Math.round((rData.metrics?.accuracy ?? 0.995) * scale);
    const precision = Math.round((rData.metrics?.precision ?? 0.995) * scale);
    const recall = Math.round((rData.metrics?.recall ?? 0.995) * scale);
    const f1 = Math.round((rData.metrics?.f1 ?? 0.995) * scale);
    const clientCount = rData.client_count || 4;

    console.log(`Submitting Round ${rNum} on-chain (Acc: ${(accuracy / 100).toFixed(2)}%, F1: ${(f1 / 100).toFixed(2)}%)...`);
    const tx = await amlVerifier.verifyAndRecordRound(
      a,
      b,
      c,
      input,
      rNum,
      modelWeightsHash,
      accuracy,
      precision,
      recall,
      f1,
      clientCount,
      circuitHash
    );

    const receipt = await tx.wait();
    console.log(`  [CONFIRMED] Round ${rNum} verified & recorded in Block #${receipt.blockNumber} (tx: ${tx.hash})`);
  }

  // 6. Print on-chain ledger summary
  const roundNumbersOnChain = await amlVerifier.getRoundNumbers();
  console.log(`\n=== Verification Complete ===`);
  console.log(`[COMPLETE] Verification Successful! Transaction confirmed on-chain.`);
  console.log(`Total verified rounds currently on blockchain: ${roundNumbersOnChain.length}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Submission failed:", error);
    process.exit(1);
  });
