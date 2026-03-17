// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract LexiCacheVerifier {
    struct VerificationRecord {
        string docHash;
        string[] clauseTypes;
        uint256 timestamp;
        string analysisHash;
        address verifier;
    }

    VerificationRecord[] private records;

    event VerificationStored(
        uint256 indexed recordId,
        string docHash,
        address indexed verifier,
        uint256 timestamp,
        string analysisHash
    );

    function storeVerification(
        string calldata docHash,
        string[] calldata clauseTypes,
        uint256 timestamp,
        string calldata analysisHash
    ) external returns (uint256 recordId) {
        records.push(
            VerificationRecord({
                docHash: docHash,
                clauseTypes: clauseTypes,
                timestamp: timestamp,
                analysisHash: analysisHash,
                verifier: msg.sender
            })
        );

        recordId = records.length - 1;
        emit VerificationStored(recordId, docHash, msg.sender, timestamp, analysisHash);
    }

    function getVerification(uint256 recordId)
        external
        view
        returns (
            string memory docHash,
            string[] memory clauseTypes,
            uint256 timestamp,
            string memory analysisHash,
            address verifier
        )
    {
        VerificationRecord storage r = records[recordId];
        return (r.docHash, r.clauseTypes, r.timestamp, r.analysisHash, r.verifier);
    }

    function totalRecords() external view returns (uint256) {
        return records.length;
    }
}
