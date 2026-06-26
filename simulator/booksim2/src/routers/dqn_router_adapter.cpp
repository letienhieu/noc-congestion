#include "routers/dqn_router_adapter.hpp"

#include <iostream>

DQNRouterAdapter::DQNRouterAdapter(const Configuration &config)
    : _warned_missing_model(false) {
  _policy.Configure(config);
}

int DQNRouterAdapter::SelectPort(const Router *router, const Flit *flit,
                                 const std::vector<int> &candidate_ports,
                                 int xy_port, int adaptive_port) const {
  if(candidate_ports.empty()) {
    return xy_port;
  }

  if(_policy.ModelReady()) {
    return _policy.SelectPort(router, flit, candidate_ports);
  }

  if(!_warned_missing_model) {
    std::cerr << "[DQN] Model assets missing or incomplete. "
              << "Using fallback policy='" << _policy.FallbackPolicy()
              << "', stub='" << _policy.StubPolicy() << "'." << std::endl;
    _warned_missing_model = true;
  }

  const std::string &fallback = _policy.FallbackPolicy();
  if(fallback == "xy") {
    return xy_port;
  }
  if(fallback == "adaptive") {
    return adaptive_port;
  }

  // fallback=stub (default)
  return _policy.SelectPort(router, flit, candidate_ports);
}
