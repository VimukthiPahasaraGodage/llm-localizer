import json

sol_code = '''pragma solidity ^0.6.0;
contract Core is Storage {
  using BytesConvert for bytes;
  modifier onlyProxy {
    require(delegates[proxyDelegateIds[msg.sender]] != address(0), "CO01");
    _;
  }
  function delegateCall(address _proxy) internal returns (bool status)
  {
    uint256 delegateId = proxyDelegateIds[_proxy];
    address delegate = delegates[delegateId];
    require(delegate != address(0), "CO02");
    (status, ) = delegate.delegatecall(msg.data);
    require(status, "CO03");
  }
  function delegateCallUint256(address _proxy)
    internal returns (uint256)
  {
    return delegateCallBytes(_proxy).toUint256();
  }
  function delegateCallBytes(address _proxy)
    internal returns (bytes memory result)
  {
    bool status;
    uint256 delegateId = proxyDelegateIds[_proxy];
    address delegate = delegates[delegateId];
    require(delegate != address(0), "CO02");
    (status, result) = delegate.delegatecall(msg.data); 
    require(status, "CO03");
  }
  function defineDelegateInternal(uint256 _delegateId, address _delegate) internal returns (bool) {
    require(_delegateId != 0, "CO04");
    delegates[_delegateId] = _delegate;
    return true;
  }
  function defineProxyInternal(address _proxy, uint256 _delegateId)
    virtual internal returns (bool)
  {
    require(delegates[_delegateId] != address(0), "CO02");
    require(_proxy != address(0), "CO05");
    proxyDelegateIds[_proxy] = _delegateId;
    return true;
  }
  function migrateProxyInternal(address _proxy, address _newCore)
    internal returns (bool)
  {
    require(proxyDelegateIds[_proxy] != 0, "CO06");
    require(Proxy(_proxy).updateCore(_newCore), "CO07");
    return true;
  }
  function removeProxyInternal(address _proxy)
    internal returns (bool)
  {
    require(proxyDelegateIds[_proxy] != 0, "CO06");
    delete proxyDelegateIds[_proxy];
    return true;
  }
}'''

payload = {
    "code": sol_code
}

print(json.dumps(payload))
