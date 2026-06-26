#include "routers/dqn_policy.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

#include "random_utils.hpp"

DQNPolicy::DQNPolicy()
    : _metadata_loaded(false),
      _weights_loaded(false),
      _mlp_ready(false),
      _model_dir(""),
      _metadata_file("metadata.json"),
      _weights_file("weights.txt"),
      _stub_policy("rule_based"),
      _fallback_policy("stub"),
      _model_type("mlp"),
      _debug(false),
      _printed_debug_decision(false),
      _nodes(1),
      _k(1),
      _n(2),
      _c(1),
      _vc_buf_size(1),
      _input_dim(10),
      _hidden1_dim(64),
      _hidden2_dim(64),
      _output_dim(2) {}

void DQNPolicy::Configure(const Configuration &config) {
  _model_dir = config.GetStr("dqn_model_dir");
  _metadata_file = config.GetStr("dqn_model_metadata");
  _weights_file = config.GetStr("dqn_weights_file");
  _stub_policy = config.GetStr("dqn_stub_policy");
  _fallback_policy = config.GetStr("dqn_fallback_policy");
  _debug = (config.GetInt("dqn_debug") > 0);

  _metadata_loaded = false;
  _weights_loaded = false;
  _mlp_ready = false;
  _printed_debug_decision = false;

  _table_policy.clear();
  _w1.clear();
  _b1.clear();
  _w2.clear();
  _b2.clear();
  _w3.clear();
  _b3.clear();

  _model_type = "mlp";
  _input_dim = 10;
  _hidden1_dim = 64;
  _hidden2_dim = 64;
  _output_dim = 2;

  _k = std::max(1, config.GetInt("k"));
  _n = std::max(1, config.GetInt("n"));
  _c = std::max(1, config.GetInt("c"));
  _nodes = 1;
  for(int i = 0; i < _n; ++i) {
    _nodes *= _k;
  }
  _nodes *= _c;
  _vc_buf_size = std::max(1, config.GetInt("vc_buf_size"));

  if(_model_dir.empty()) {
    return;
  }

  const std::string metadata_path = _JoinPath(_model_dir, _metadata_file);
  const std::string weights_path = _JoinPath(_model_dir, _weights_file);

  _metadata_loaded = _FileExists(metadata_path);
  _weights_loaded = _FileExists(weights_path);

  if(_metadata_loaded) {
    _LoadMetadataHints();
  }

  if(_weights_loaded) {
    _LoadTableWeights();
    _LoadMLPWeights();
  }

  const size_t expected_w1 = static_cast<size_t>(_input_dim) * static_cast<size_t>(_hidden1_dim);
  const size_t expected_b1 = static_cast<size_t>(_hidden1_dim);
  const size_t expected_w2 = static_cast<size_t>(_hidden1_dim) * static_cast<size_t>(_hidden2_dim);
  const size_t expected_b2 = static_cast<size_t>(_hidden2_dim);
  const size_t expected_w3 = static_cast<size_t>(_hidden2_dim) * static_cast<size_t>(_output_dim);
  const size_t expected_b3 = static_cast<size_t>(_output_dim);

  _mlp_ready = _metadata_loaded && _weights_loaded &&
               (_model_type == "mlp") &&
               (_w1.size() == expected_w1) && (_b1.size() == expected_b1) &&
               (_w2.size() == expected_w2) && (_b2.size() == expected_b2) &&
               (_w3.size() == expected_w3) && (_b3.size() == expected_b3);


  // Experience logging
  _experience_log_path = "";
  _experience_count = 0;
  {
    const std::string exp_log = config.GetStr("dqn_experience_log");
    if(!exp_log.empty() && exp_log != "none") {
      _experience_log_path = exp_log;
      _experience_log.open(exp_log.c_str());
      if(_experience_log.good()) {
        // Write CSV header
        _experience_log << "router_id,flit_id,flit_src,flit_dest,flit_hops,";
        for(int i = 0; i < _input_dim; ++i) {
          _experience_log << "s" << i << ",";
        }
        _experience_log << "n_candidates,";
        for(int i = 0; i < 5; ++i) {
          _experience_log << "cand" << i << ",";
        }
        _experience_log << "action_index,selected_port,";
        for(int i = 0; i < 5; ++i) {
          _experience_log << "q" << i << ",";
        }
        _experience_log << "reason" << std::endl;
      }
    }
  }

  if(_debug) {
    std::cerr << "[DQN] configure model_dir=" << _model_dir
              << " metadata=" << (_metadata_loaded ? "yes" : "no")
              << " weights=" << (_weights_loaded ? "yes" : "no")
              << " mlp_ready=" << (_mlp_ready ? "yes" : "no")
              << " dims=" << _input_dim << "x" << _hidden1_dim
              << "x" << _hidden2_dim << "x" << _output_dim
              << " stub_policy=" << _stub_policy
              << " fallback_policy=" << _fallback_policy << std::endl;
  }
}

int DQNPolicy::SelectPort(const Router *router, const Flit *flit,
                          const std::vector<int> &candidate_ports) const {
  if(candidate_ports.empty()) {
    return -1;
  }

  if(_mlp_ready) {
    if(candidate_ports.size() > static_cast<size_t>(_output_dim)) {
      int fallback_port = _SelectStub(router, flit, candidate_ports);
      _DebugDecision(router, flit, candidate_ports, std::vector<double>(),
                     std::vector<double>(), -1, fallback_port,
                     "mlp_output_too_small_fallback_stub");
      return fallback_port;
    }

    std::vector<double> state = _BuildState(router, flit, candidate_ports);
    std::vector<double> logits = _RunMLP(state);
    int action_index = _SelectMaskedAction(logits, candidate_ports.size());

    if((action_index >= 0) &&
       (action_index < static_cast<int>(candidate_ports.size()))) {
      int selected_port = candidate_ports[action_index];
      _LogExperience(router, flit, candidate_ports, state, action_index, selected_port);
      _DebugDecision(router, flit, candidate_ports, state, logits,
                     action_index, selected_port, "mlp");
      return selected_port;
    }

    int fallback_port = _SelectStub(router, flit, candidate_ports);
    _DebugDecision(router, flit, candidate_ports, state, logits,
                   action_index, fallback_port,
                   "invalid_masked_action_fallback_stub");
    return fallback_port;
  }

  int stub_port = _SelectStub(router, flit, candidate_ports);
  _DebugDecision(router, flit, candidate_ports, std::vector<double>(),
                 std::vector<double>(), -1, stub_port,
                 "mlp_not_ready_stub");
  return stub_port;
}

bool DQNPolicy::ModelReady() const {
  return _mlp_ready;
}

const std::string &DQNPolicy::StubPolicy() const {
  return _stub_policy;
}

const std::string &DQNPolicy::FallbackPolicy() const {
  return _fallback_policy;
}

bool DQNPolicy::_FileExists(const std::string &path) const {
  std::ifstream in(path.c_str());
  return in.good();
}

std::string DQNPolicy::_JoinPath(const std::string &base, const std::string &leaf) const {
  if(base.empty()) {
    return leaf;
  }
  if(leaf.empty()) {
    return base;
  }
  if(base[base.size() - 1] == '/') {
    return base + leaf;
  }
  return base + "/" + leaf;
}

void DQNPolicy::_LoadMetadataHints() {
  const std::string metadata_path = _JoinPath(_model_dir, _metadata_file);
  std::ifstream in(metadata_path.c_str());
  if(!in.good()) {
    return;
  }

  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string content = buffer.str();

  std::string value;
  if(_TryExtractValue(content, "stub_policy", &value) && !value.empty()) {
    _stub_policy = value;
  }
  if(_TryExtractValue(content, "fallback_policy", &value) && !value.empty()) {
    _fallback_policy = value;
  }
  if(_TryExtractValue(content, "model_type", &value) && !value.empty()) {
    _model_type = value;
  }

  int parsed_int = 0;
  if(_TryExtractValue(content, "input_dim", &value) && _TryParseInt(value, &parsed_int) &&
     (parsed_int > 0)) {
    _input_dim = parsed_int;
  }
  if(_TryExtractValue(content, "hidden1_dim", &value) && _TryParseInt(value, &parsed_int) &&
     (parsed_int > 0)) {
    _hidden1_dim = parsed_int;
  }
  if(_TryExtractValue(content, "hidden2_dim", &value) && _TryParseInt(value, &parsed_int) &&
     (parsed_int > 0)) {
    _hidden2_dim = parsed_int;
  }
  if(_TryExtractValue(content, "output_dim", &value) && _TryParseInt(value, &parsed_int) &&
     (parsed_int > 0)) {
    _output_dim = parsed_int;
  }
}

void DQNPolicy::_LoadTableWeights() {
  const std::string weights_path = _JoinPath(_model_dir, _weights_file);
  std::ifstream in(weights_path.c_str());
  if(!in.good()) {
    return;
  }

  std::string line;
  while(std::getline(in, line)) {
    if(line.empty()) {
      continue;
    }
    if(line[0] == '#') {
      continue;
    }

    std::replace(line.begin(), line.end(), ',', ' ');
    std::stringstream ss(line);
    int router_id = -1;
    int dest_id = -1;
    int out_port = -1;
    if(ss >> router_id >> dest_id >> out_port) {
      _table_policy[std::make_pair(router_id, dest_id)] = out_port;
    }
  }
}

void DQNPolicy::_LoadMLPWeights() {
  const std::string weights_path = _JoinPath(_model_dir, _weights_file);
  std::ifstream in(weights_path.c_str());
  if(!in.good()) {
    return;
  }

  std::vector<double> values;
  std::string line;
  while(std::getline(in, line)) {
    size_t comment_pos = line.find('#');
    if(comment_pos != std::string::npos) {
      line = line.substr(0, comment_pos);
    }

    std::replace(line.begin(), line.end(), ',', ' ');
    std::stringstream ss(line);
    std::string token;
    while(ss >> token) {
      char *endptr = NULL;
      double parsed = std::strtod(token.c_str(), &endptr);
      if(endptr && (*endptr == '\0') && (endptr != token.c_str())) {
        values.push_back(parsed);
      }
    }
  }

  const size_t expected_w1 = static_cast<size_t>(_input_dim) * static_cast<size_t>(_hidden1_dim);
  const size_t expected_b1 = static_cast<size_t>(_hidden1_dim);
  const size_t expected_w2 = static_cast<size_t>(_hidden1_dim) * static_cast<size_t>(_hidden2_dim);
  const size_t expected_b2 = static_cast<size_t>(_hidden2_dim);
  const size_t expected_w3 = static_cast<size_t>(_hidden2_dim) * static_cast<size_t>(_output_dim);
  const size_t expected_b3 = static_cast<size_t>(_output_dim);
  const size_t expected_total = expected_w1 + expected_b1 + expected_w2 + expected_b2 + expected_w3 + expected_b3;

  if(values.size() != expected_total) {
    if(_debug) {
      std::cerr << "[DQN] MLP weight parse mismatch: expected " << expected_total
                << " numeric values, got " << values.size() << std::endl;
    }
    return;
  }

  size_t offset = 0;
  _w1.assign(values.begin() + offset, values.begin() + offset + expected_w1);
  offset += expected_w1;
  _b1.assign(values.begin() + offset, values.begin() + offset + expected_b1);
  offset += expected_b1;
  _w2.assign(values.begin() + offset, values.begin() + offset + expected_w2);
  offset += expected_w2;
  _b2.assign(values.begin() + offset, values.begin() + offset + expected_b2);
  offset += expected_b2;
  _w3.assign(values.begin() + offset, values.begin() + offset + expected_w3);
  offset += expected_w3;
  _b3.assign(values.begin() + offset, values.begin() + offset + expected_b3);
}

bool DQNPolicy::_TryExtractValue(const std::string &content, const std::string &key,
                                 std::string *value) const {
  const std::string quoted_key = "\"" + key + "\"";
  size_t pos = content.find(quoted_key);
  if(pos == std::string::npos) {
    pos = content.find(key);
    if(pos == std::string::npos) {
      return false;
    }
  }

  size_t sep = content.find(':', pos);
  if(sep == std::string::npos) {
    sep = content.find('=', pos);
    if(sep == std::string::npos) {
      return false;
    }
  }

  size_t start = sep + 1;
  while(start < content.size() && std::isspace(static_cast<unsigned char>(content[start]))) {
    ++start;
  }

  if(start >= content.size()) {
    return false;
  }

  if(content[start] == '"') {
    ++start;
    size_t end = content.find('"', start);
    if(end == std::string::npos) {
      return false;
    }
    *value = content.substr(start, end - start);
    return true;
  }

  size_t end = start;
  while(end < content.size() && content[end] != '\n' && content[end] != ',' && content[end] != '}') {
    ++end;
  }
  *value = content.substr(start, end - start);
  while(!value->empty() && std::isspace(static_cast<unsigned char>((*value)[value->size() - 1]))) {
    value->erase(value->size() - 1);
  }
  return !value->empty();
}

bool DQNPolicy::_TryParseInt(const std::string &text, int *value) const {
  if(!value) {
    return false;
  }
  char *endptr = NULL;
  long parsed = std::strtol(text.c_str(), &endptr, 10);
  if((endptr == text.c_str()) || (endptr == NULL) || (*endptr != '\0')) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

int DQNPolicy::_SelectStub(const Router *router, const Flit *flit,
                           const std::vector<int> &candidates) const {
  if(_weights_loaded && (_stub_policy == "table")) {
    int table_port = _SelectTable(router, flit, candidates);
    if(table_port >= 0) {
      return table_port;
    }
  }

  if(_stub_policy == "random") {
    return _SelectRandom(candidates);
  }

  return _SelectRuleBased(router, candidates);
}

int DQNPolicy::_SelectRuleBased(const Router *router, const std::vector<int> &candidates) const {
  int best_port = candidates[0];
  int best_credit = router->GetUsedCredit(best_port);
  for(size_t i = 1; i < candidates.size(); ++i) {
    const int port = candidates[i];
    const int used_credit = router->GetUsedCredit(port);
    if(used_credit < best_credit) {
      best_credit = used_credit;
      best_port = port;
    }
  }
  return best_port;
}

int DQNPolicy::_SelectRandom(const std::vector<int> &candidates) const {
  if(candidates.size() == 1) {
    return candidates[0];
  }
  const int idx = RandomInt(static_cast<int>(candidates.size()) - 1);
  return candidates[idx];
}

int DQNPolicy::_SelectTable(const Router *router, const Flit *flit,
                            const std::vector<int> &candidates) const {
  const std::pair<int, int> key(router->GetID(), flit->dest);
  std::map<std::pair<int, int>, int>::const_iterator iter = _table_policy.find(key);
  if(iter == _table_policy.end()) {
    return -1;
  }
  const int candidate = iter->second;
  for(size_t i = 0; i < candidates.size(); ++i) {
    if(candidates[i] == candidate) {
      return candidate;
    }
  }
  return -1;
}


void DQNPolicy::_LogExperience(const Router *router, const Flit *flit,
                               const std::vector<int> &candidate_ports,
                               const std::vector<double> &state,
                               int action_index, int selected_port) const {
  if(!_experience_log.good()) return;

  _experience_log << router->GetID() << ","
                  << flit->id << ","
                  << flit->src << ","
                  << flit->dest << ","
                  << flit->hops << ",";

  // State vector
  for(size_t i = 0; i < state.size(); ++i) {
    _experience_log << state[i] << ",";
  }

  // Candidate ports (pad to 5)
  _experience_log << candidate_ports.size() << ",";
  for(int i = 0; i < 5; ++i) {
    if(i < static_cast<int>(candidate_ports.size())) {
      _experience_log << candidate_ports[i] << ",";
    } else {
      _experience_log << -1 << ",";
    }
  }

  _experience_log << action_index << "," << selected_port << ",";

  // Q-values from last MLP run (recompute for logging)
  std::vector<double> logits = _RunMLP(state);
  for(int i = 0; i < 5; ++i) {
    if(i < static_cast<int>(logits.size())) {
      _experience_log << logits[i] << ",";
    } else {
      _experience_log << 0.0 << ",";
    }
  }

  _experience_log << "mlp" << std::endl;
  ++_experience_count;
}

std::vector<double> DQNPolicy::_BuildState(const Router *router, const Flit *flit,
                                           const std::vector<int> &candidate_ports) const {
  // State vector layout (default first 10 features):
  // 0: router_id_norm
  // 1: dest_id_norm
  // 2: same_router_flag
  // 3: flit_class_norm
  // 4: flit_type_norm
  // 5: candidate_count_norm
  // 6: candidate_port_0_norm
  // 7: candidate_port_1_norm
  // 8: candidate_credit_0_norm
  // 9: candidate_credit_1_norm
  // Additional dimensions (if input_dim > 10) are zero-padded, except:
  // 10: credit_delta_norm (if present)
  // 11: hop_count_norm (if present)
  // 12: src_id_norm (if present)
  // 13: rem_x_norm (if present)
  // 14: rem_y_norm (if present)
  // 15: axis_pref_y_minus_x (if present)
  // 16: hash_balance_feature (if present)
  // 17: local_output_pressure_norm (if present)
  // 18: candidate_credit_min_norm (if present)
  // 19: candidate_credit_spread_norm (if present)
  std::vector<double> state(static_cast<size_t>(_input_dim), 0.0);

  const double node_den = std::max(1, _nodes - 1);
  const double out_den = std::max(1, router->NumOutputs() - 1);

  if(_input_dim > 0) {
    state[0] = static_cast<double>(router->GetID()) / node_den;
  }
  if(_input_dim > 1) {
    state[1] = static_cast<double>(flit->dest) / node_den;
  }
  if(_input_dim > 2) {
    state[2] = (router->GetID() == flit->dest) ? 1.0 : 0.0;
  }
  if(_input_dim > 3) {
    state[3] = static_cast<double>(flit->cl) / 8.0;
  }
  if(_input_dim > 4) {
    state[4] = static_cast<double>(flit->type) / static_cast<double>(Flit::NUM_FLIT_TYPES - 1);
  }
  if(_input_dim > 5) {
    state[5] = static_cast<double>(candidate_ports.size()) /
               static_cast<double>(std::max(1, _output_dim));
  }

  int c0_port = -1;
  int c1_port = -1;
  int c0_credit = -1;
  int c1_credit = -1;

  if(!candidate_ports.empty()) {
    c0_port = candidate_ports[0];
    c0_credit = router->GetUsedCredit(c0_port);
  }
  if(candidate_ports.size() > 1) {
    c1_port = candidate_ports[1];
    c1_credit = router->GetUsedCredit(c1_port);
  }

  if(_input_dim > 6) {
    state[6] = (c0_port >= 0) ? (static_cast<double>(c0_port) / out_den) : -1.0;
  }
  if(_input_dim > 7) {
    state[7] = (c1_port >= 0) ? (static_cast<double>(c1_port) / out_den) : -1.0;
  }
  if(_input_dim > 8) {
    state[8] = (c0_credit >= 0) ? (static_cast<double>(c0_credit) / _vc_buf_size) : -1.0;
  }
  if(_input_dim > 9) {
    state[9] = (c1_credit >= 0) ? (static_cast<double>(c1_credit) / _vc_buf_size) : -1.0;
  }
  if(_input_dim > 10) {
    double delta = 0.0;
    if((c0_credit >= 0) && (c1_credit >= 0)) {
      delta = static_cast<double>(c0_credit - c1_credit) /
              static_cast<double>(std::max(1, _vc_buf_size));
    }
    state[10] = delta;
  }
  if(_input_dim > 11) {
    state[11] = static_cast<double>(flit->hops) / 32.0;
  }
  if(_input_dim > 12) {
    state[12] = static_cast<double>(flit->src) / node_den;
  }
  if(_input_dim > 13) {
    const int cur_node = router->GetID() / std::max(1, _c);
    const int dst_node = flit->dest / std::max(1, _c);
    const int dx = std::abs(_CoordX(dst_node) - _CoordX(cur_node));
    const double den = static_cast<double>(std::max(1, _k - 1));
    state[13] = static_cast<double>(dx) / den;
  }
  if(_input_dim > 14) {
    const int cur_node = router->GetID() / std::max(1, _c);
    const int dst_node = flit->dest / std::max(1, _c);
    const int dy = std::abs(_CoordY(dst_node) - _CoordY(cur_node));
    const double den = static_cast<double>(std::max(1, _k - 1));
    state[14] = static_cast<double>(dy) / den;
  }
  if(_input_dim > 15) {
    state[15] = state[14] - state[13];
  }
  if(_input_dim > 16) {
    state[16] = _HashBalanceFeature(flit);
  }
  if(_input_dim > 17) {
    const int num_out = std::max(1, router->NumOutputs());
    int total_credit = 0;
    int samples = 0;
    for(int p = 0; p < num_out; ++p) {
      total_credit += std::max(0, router->GetUsedCredit(p));
      ++samples;
    }
    state[17] = (samples > 0)
                    ? (static_cast<double>(total_credit) /
                       static_cast<double>(samples * std::max(1, _vc_buf_size)))
                    : 0.0;
  }
  if(_input_dim > 18) {
    double c0 = (c0_credit >= 0) ? static_cast<double>(c0_credit) : static_cast<double>(_vc_buf_size);
    double c1 = (c1_credit >= 0) ? static_cast<double>(c1_credit) : static_cast<double>(_vc_buf_size);
    state[18] = std::min(c0, c1) / static_cast<double>(std::max(1, _vc_buf_size));
  }
  if(_input_dim > 19) {
    double c0 = (c0_credit >= 0) ? static_cast<double>(c0_credit) : static_cast<double>(_vc_buf_size);
    double c1 = (c1_credit >= 0) ? static_cast<double>(c1_credit) : static_cast<double>(_vc_buf_size);
    state[19] = std::fabs(c0 - c1) / static_cast<double>(std::max(1, _vc_buf_size));
  }

  return state;
}

int DQNPolicy::_CoordX(int node_id) const {
  const int k = std::max(1, _k);
  return node_id % k;
}

int DQNPolicy::_CoordY(int node_id) const {
  const int k = std::max(1, _k);
  return (node_id / k) % k;
}

double DQNPolicy::_HashBalanceFeature(const Flit *flit) const {
  uint32_t s = static_cast<uint32_t>(std::max(0, flit->src));
  uint32_t d = static_cast<uint32_t>(std::max(0, flit->dest));
  uint32_t id = static_cast<uint32_t>(std::max(0, flit->id));
  uint32_t mix = s * 2654435761u;
  mix ^= (d + 0x9e3779b9u + (mix << 6) + (mix >> 2));
  mix ^= (id + 0x85ebca6bu + (mix << 6) + (mix >> 2));
  return (mix & 0x1u) ? 1.0 : -1.0;
}

std::vector<double> DQNPolicy::_RunMLP(const std::vector<double> &state) const {
  if((static_cast<int>(state.size()) != _input_dim) || !_mlp_ready) {
    return std::vector<double>();
  }

  std::vector<double> h1(static_cast<size_t>(_hidden1_dim), 0.0);
  for(int o = 0; o < _hidden1_dim; ++o) {
    double acc = _b1[static_cast<size_t>(o)];
    const size_t row_offset = static_cast<size_t>(o) * static_cast<size_t>(_input_dim);
    for(int i = 0; i < _input_dim; ++i) {
      acc += _w1[row_offset + static_cast<size_t>(i)] * state[static_cast<size_t>(i)];
    }
    h1[static_cast<size_t>(o)] = std::max(0.0, acc);
  }

  std::vector<double> h2(static_cast<size_t>(_hidden2_dim), 0.0);
  for(int o = 0; o < _hidden2_dim; ++o) {
    double acc = _b2[static_cast<size_t>(o)];
    const size_t row_offset = static_cast<size_t>(o) * static_cast<size_t>(_hidden1_dim);
    for(int i = 0; i < _hidden1_dim; ++i) {
      acc += _w2[row_offset + static_cast<size_t>(i)] * h1[static_cast<size_t>(i)];
    }
    h2[static_cast<size_t>(o)] = std::max(0.0, acc);
  }

  std::vector<double> logits(static_cast<size_t>(_output_dim), 0.0);
  for(int o = 0; o < _output_dim; ++o) {
    double acc = _b3[static_cast<size_t>(o)];
    const size_t row_offset = static_cast<size_t>(o) * static_cast<size_t>(_hidden2_dim);
    for(int i = 0; i < _hidden2_dim; ++i) {
      acc += _w3[row_offset + static_cast<size_t>(i)] * h2[static_cast<size_t>(i)];
    }
    logits[static_cast<size_t>(o)] = acc;
  }

  return logits;
}

int DQNPolicy::_SelectMaskedAction(const std::vector<double> &logits,
                                   size_t valid_action_count) const {
  if(logits.empty() || (valid_action_count == 0)) {
    return -1;
  }

  const double negative_inf = -std::numeric_limits<double>::infinity();
  int best_index = -1;
  double best_score = negative_inf;

  // Valid-action masking: only logits for action index < valid_action_count
  // can be selected. Invalid actions are forced to -inf.
  for(size_t i = 0; i < logits.size(); ++i) {
    const bool valid = (i < valid_action_count);
    const double masked_score = valid ? logits[i] : negative_inf;
    if(masked_score > best_score) {
      best_score = masked_score;
      best_index = static_cast<int>(i);
    }
  }

  if((best_index < 0) || (best_score == negative_inf)) {
    return -1;
  }
  return best_index;
}

void DQNPolicy::_DebugDecision(const Router *router, const Flit *flit,
                               const std::vector<int> &candidate_ports,
                               const std::vector<double> &state,
                               const std::vector<double> &logits,
                               int action_index,
                               int selected_port,
                               const std::string &reason) const {
  if(!_debug || _printed_debug_decision) {
    return;
  }

  _printed_debug_decision = true;

  std::cerr << "[DQN][debug] One decision sample" << std::endl;
  std::cerr << "  reason=" << reason
            << " router=" << router->GetID()
            << " dest=" << flit->dest
            << " action_index=" << action_index
            << " selected_port=" << selected_port << std::endl;

  std::cerr << "  candidates=[";
  for(size_t i = 0; i < candidate_ports.size(); ++i) {
    if(i) {
      std::cerr << ",";
    }
    std::cerr << candidate_ports[i];
  }
  std::cerr << "]" << std::endl;

  if(!state.empty()) {
    std::cerr << "  state=[";
    for(size_t i = 0; i < state.size(); ++i) {
      if(i) {
        std::cerr << ",";
      }
      std::cerr << state[i];
    }
    std::cerr << "]" << std::endl;
  }

  if(!logits.empty()) {
    std::cerr << "  logits=[";
    for(size_t i = 0; i < logits.size(); ++i) {
      if(i) {
        std::cerr << ",";
      }
      std::cerr << logits[i];
    }
    std::cerr << "]" << std::endl;
  }
}
