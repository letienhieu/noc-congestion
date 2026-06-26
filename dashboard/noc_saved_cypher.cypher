// NoC STORM - Neo4j Browser saved queries (Mesh 4x4 dataset, bolt://localhost:7688)

// ===== 1. Paper figures (data) =====

// Fig neo4j_schema - two-layer schema
CALL db.schema.visualization();

// Fig neo4j_instance - instance graph (router 5 + state)
MATCH (r:Router {mesh_id:'mesh_4x4', id:5}) OPTIONAL MATCH (r)-[l:LINK]->(nb:Router) OPTIONAL MATCH (r)<-[o:OBSERVED_AT]-(rs:RouterState)-[ins:IN_SNAPSHOT]->(s:Snapshot {run_id:'mesh_4x4_hotspot_ir050'}) WHERE s.sample_idx < 2 RETURN r,l,nb,o,rs,ins,s;

// Fig heatmap_actual - actual congestion grid (t=30)
MATCH (s:Snapshot {run_id:'mesh_4x4_hotspot_ir050', sample_idx:30})<-[:IN_SNAPSHOT]-(rs:RouterState) MATCH (r:Router {mesh_id:'mesh_4x4', id:rs.router_id}) RETURN r.y AS row, r.x AS col, rs.router_id AS router, round(rs.buffer_occupancy_norm,4) AS occupancy ORDER BY row,col;

// Fig preds_vs_actual - router 5 actual time series
MATCH (rs:RouterState {run_id:'mesh_4x4_hotspot_ir050', router_id:5}) RETURN rs.sample_idx AS t, round(rs.buffer_occupancy_norm,4) AS actual_occupancy ORDER BY t;

// ===== 2. Topology & schema =====

// Node counts by label
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC;

// Relationship counts by type
MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS count ORDER BY count DESC;

// Full 4x4 mesh (Router-LINK)
MATCH (a:Router {mesh_id:'mesh_4x4'})-[l:LINK]->(b:Router) RETURN a,l,b;

// Router coordinates + out-degree
MATCH (r:Router {mesh_id:'mesh_4x4'}) RETURN r.id AS router, r.x AS x, r.y AS y, size([(r)-[:LINK]->()|1]) AS out_degree ORDER BY router;

// Neighbors of router 5
MATCH (r:Router {mesh_id:'mesh_4x4', id:5})-[:LINK]-(nb:Router) RETURN DISTINCT nb.id AS neighbor, nb.x AS x, nb.y AS y ORDER BY neighbor;

// ===== 3. Congestion analysis (showcase hotspot ir050) =====

// Congestion curve over time (mean per snapshot)
MATCH (rs:RouterState {run_id:'mesh_4x4_hotspot_ir050'}) RETURN rs.sample_idx AS t, round(avg(rs.buffer_occupancy_norm),4) AS mean_occ ORDER BY t;

// Peak snapshot (highest total occupancy)
MATCH (rs:RouterState {run_id:'mesh_4x4_hotspot_ir050'}) RETURN rs.sample_idx AS t, round(sum(rs.buffer_occupancy_norm),3) AS total_occ ORDER BY total_occ DESC LIMIT 5;

// Top 10 most-congested (router, time)
MATCH (rs:RouterState {run_id:'mesh_4x4_hotspot_ir050'}) RETURN rs.router_id AS router, rs.sample_idx AS t, round(rs.buffer_occupancy_norm,4) AS occ ORDER BY occ DESC LIMIT 10;

// Average occupancy per router (hotspot map)
MATCH (rs:RouterState {run_id:'mesh_4x4_hotspot_ir050'}) RETURN rs.router_id AS router, round(avg(rs.buffer_occupancy_norm),4) AS avg_occ ORDER BY avg_occ DESC;

// Max occupancy per router
MATCH (rs:RouterState {run_id:'mesh_4x4_hotspot_ir050'}) RETURN rs.router_id AS router, round(max(rs.buffer_occupancy_norm),4) AS max_occ ORDER BY max_occ DESC;

// ===== 4. Runs & traffic patterns =====

// All runs + snapshot counts
MATCH (s:Snapshot) RETURN s.run_id AS run, count(s) AS snapshots ORDER BY run;

// Mean occupancy per run
MATCH (rs:RouterState) RETURN rs.run_id AS run, round(avg(rs.buffer_occupancy_norm),4) AS mean_occ ORDER BY mean_occ DESC;

// Mean occupancy vs injection rate (hotspot)
MATCH (rs:RouterState) WHERE rs.run_id STARTS WITH 'mesh_4x4_hotspot_' RETURN rs.run_id AS run, round(avg(rs.buffer_occupancy_norm),4) AS mean_occ ORDER BY run;

// Compare 4 patterns at ir050
MATCH (rs:RouterState) WHERE rs.run_id STARTS WITH 'mesh_4x4_' AND rs.run_id ENDS WITH '_ir050' RETURN rs.run_id AS run, round(avg(rs.buffer_occupancy_norm),4) AS mean_occ ORDER BY mean_occ DESC;

// Throughput per run (avg received/sent)
MATCH (rs:RouterState) RETURN rs.run_id AS run, round(avg(rs.received_total),1) AS avg_received, round(avg(rs.sent_total),1) AS avg_sent ORDER BY run;

// ===== 5. Dynamics & extension (future work) =====

// Snapshot NEXT temporal chain (first 12)
MATCH (s:Snapshot {run_id:'mesh_4x4_hotspot_ir050'})-[:NEXT]->(s2:Snapshot) RETURN s.sample_idx AS t, s2.sample_idx AS t_next, s.cycle_start AS cyc ORDER BY t LIMIT 12;

// Neighbor congestion correlation (t=30)
MATCH (r:Router {mesh_id:'mesh_4x4'})-[:LINK]->(nb:Router) MATCH (rs:RouterState {run_id:'mesh_4x4_hotspot_ir050', sample_idx:30, router_id:r.id}) MATCH (rsn:RouterState {run_id:'mesh_4x4_hotspot_ir050', sample_idx:30, router_id:nb.id}) RETURN r.id AS router, round(rs.buffer_occupancy_norm,4) AS occ, round(avg(rsn.buffer_occupancy_norm),4) AS neighbor_avg ORDER BY occ DESC;

// Persistently congested routers (avg & max)
MATCH (rs:RouterState {run_id:'mesh_4x4_hotspot_ir050'}) RETURN rs.router_id AS router, round(avg(rs.buffer_occupancy_norm),4) AS avg_occ, round(max(rs.buffer_occupancy_norm),4) AS max_occ ORDER BY avg_occ DESC LIMIT 8;

// Injection vs ejection per router (showcase)
MATCH (rs:RouterState {run_id:'mesh_4x4_hotspot_ir050'}) RETURN rs.router_id AS router, round(avg(rs.injected),2) AS avg_injected, round(avg(rs.ejected),2) AS avg_ejected ORDER BY avg_injected DESC;

// Hotspot location across all hotspot rates
MATCH (rs:RouterState) WHERE rs.run_id STARTS WITH 'mesh_4x4_hotspot_' RETURN rs.router_id AS router, round(avg(rs.buffer_occupancy_norm),4) AS avg_occ ORDER BY avg_occ DESC LIMIT 8;
