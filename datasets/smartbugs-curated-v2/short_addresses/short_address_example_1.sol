/*
 * @vulnerable_at_lines: 16
 */

pragma solidity ^0.4.24;

contract ShortAddressVulnerable1 {
    mapping(address => uint256) public balances;

    event Transfer(address indexed from, address indexed to, uint256 value);

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
        emit Transfer(msg.sender, to, amount);
    }

    function getBalance(address user) public view returns (uint256) {
        return balances[user];
    }
}