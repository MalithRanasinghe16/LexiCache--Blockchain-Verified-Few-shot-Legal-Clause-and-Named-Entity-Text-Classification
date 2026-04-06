// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract LexiCacheVerifier {
    struct VerificationRecord {
        string docHash;
        string[] clauseTypes;
        uint256 timestamp;
        string analysisHash;
        string cid;          // IPFS Content Identifier for the pinned analysis JSON
        address verifier;
    }
    mapping(bytes32 => bool) public logged;

    VerificationRecord[] private records;

    event VerificationStored(
        uint256 indexed recordId,
        string docHash,
        address indexed verifier,
        uint256 timestamp,
        string analysisHash,
        string cid
    );

    function storeVerification(
        string calldata docHash,
        string[] calldata clauseTypes,
        uint256 timestamp,
        string calldata analysisHash,
        string calldata cid
    ) external returns (uint256 recordId) {
        bytes32 hashKey = keccak256(abi.encodePacked(analysisHash));

        // Idempotency check — revert if this exact analysis snapshot was already verified.
        // Allows the same document to be verified multiple times as users teach unknown
        // clauses (each teach cycle produces a new analysisHash). Blocks re-verification
        // when nothing new has been contributed (same analysisHash = same knowledge state).
        require(!logged[hashKey], "LexiCache: analysis already verified on-chain");

        // Mark logged before the storage write (checks-effects-interactions pattern).
        logged[hashKey] = true;

        records.push(
            VerificationRecord({
                docHash: docHash,
                clauseTypes: clauseTypes,
                timestamp: timestamp,
                analysisHash: analysisHash,
                cid: cid,
                verifier: msg.sender
            })
        );

        recordId = records.length - 1;
        emit VerificationStored(recordId, docHash, msg.sender, timestamp, analysisHash, cid);
    }

    function getVerification(uint256 recordId)
        external
        view
        returns (
            string memory docHash,
            string[] memory clauseTypes,
            uint256 timestamp,
            string memory analysisHash,
            string memory cid,
            address verifier
        )
    {
        VerificationRecord storage r = records[recordId];
        return (r.docHash, r.clauseTypes, r.timestamp, r.analysisHash, r.cid, r.verifier);
    }

    /// @notice Check whether an analysis snapshot has already been logged on-chain.
    function isLogged(string calldata analysisHash) external view returns (bool) {
        return logged[keccak256(abi.encodePacked(analysisHash))];
    }

    function totalRecords() external view returns (uint256) {
        return records.length;
    }
}
