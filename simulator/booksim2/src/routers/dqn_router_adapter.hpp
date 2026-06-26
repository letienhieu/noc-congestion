#ifndef _DQN_ROUTER_ADAPTER_HPP_
#define _DQN_ROUTER_ADAPTER_HPP_

#include <string>
#include <vector>

#include "config_utils.hpp"
#include "flit.hpp"
#include "routers/dqn_policy.hpp"
#include "routers/router.hpp"

class DQNRouterAdapter {
public:
  explicit DQNRouterAdapter(const Configuration &config);

  int SelectPort(const Router *router, const Flit *flit,
                 const std::vector<int> &candidate_ports,
                 int xy_port, int adaptive_port) const;

private:
  DQNPolicy _policy;
  mutable bool _warned_missing_model;
};

#endif
