// DQN routing with lightweight MLP inference for BookSim mesh routing.

#ifndef _DQN_POLICY_HPP_
#define _DQN_POLICY_HPP_

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "config_utils.hpp"
#include "flit.hpp"
#include "router.hpp"

class DQNPolicy {
public:
  DQNPolicy();

  void Configure(const Configuration &config);

  // Choose one port from a pre-validated candidate set.
  int SelectPort(const Router *router, const Flit *flit,
                 const std::vector<int> &candidate_ports) const;

  // True only when a complete MLP model is loaded and ready for inference.
  bool ModelReady() const;
  const std::string &StubPolicy() const;
  const std::string &FallbackPolicy() const;

private:
  bool _metadata_loaded;
  bool _weights_loaded;
  bool _mlp_ready;

  std::string _model_dir;
  std::string _metadata_file;
  std::string _weights_file;
  std::string _stub_policy;
  std::string _fallback_policy;
  std::string _model_type;
  bool _debug;
  mutable bool _printed_debug_decision;

  int _nodes;
  int _k;
  int _n;
  int _c;
  int _vc_buf_size;

  int _input_dim;
  int _hidden1_dim;
  int _hidden2_dim;
  int _output_dim;

  std::vector<double> _w1;
  std::vector<double> _b1;
  std::vector<double> _w2;
  std::vector<double> _b2;
  std::vector<double> _w3;
  std::vector<double> _b3;

  // Legacy scaffold table policy (kept for fallback compatibility).
  std::map<std::pair<int, int>, int> _table_policy;

  bool _FileExists(const std::string &path) const;
  std::string _JoinPath(const std::string &base, const std::string &leaf) const;

  void _LoadMetadataHints();
  void _LoadTableWeights();
  void _LoadMLPWeights();

  bool _TryExtractValue(const std::string &content, const std::string &key,
                        std::string *value) const;
  bool _TryParseInt(const std::string &text, int *value) const;

  int _SelectStub(const Router *router, const Flit *flit,
                  const std::vector<int> &candidates) const;
  int _SelectRuleBased(const Router *router, const std::vector<int> &candidates) const;
  int _SelectRandom(const std::vector<int> &candidates) const;
  int _SelectTable(const Router *router, const Flit *flit,
                   const std::vector<int> &candidates) const;

  std::vector<double> _BuildState(const Router *router, const Flit *flit,
                                  const std::vector<int> &candidate_ports) const;
  int _CoordX(int node_id) const;
  int _CoordY(int node_id) const;
  double _HashBalanceFeature(const Flit *flit) const;
  std::vector<double> _RunMLP(const std::vector<double> &state) const;
  int _SelectMaskedAction(const std::vector<double> &logits,
                          size_t valid_action_count) const;

  // Experience logging for online training
  std::string _experience_log_path;
  mutable std::ofstream _experience_log;
  mutable int _experience_count;
  void _LogExperience(const Router *router, const Flit *flit,
                      const std::vector<int> &candidate_ports,
                      const std::vector<double> &state,
                      int action_index, int selected_port) const;
  void _DebugDecision(const Router *router, const Flit *flit,
                      const std::vector<int> &candidate_ports,
                      const std::vector<double> &state,
                      const std::vector<double> &logits,
                      int action_index,
                      int selected_port,
                      const std::string &reason) const;
};

#endif
