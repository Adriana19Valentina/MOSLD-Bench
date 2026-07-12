import pickle

with open('./turkish_cl_outputs_1/test_1_results.pkl', 'rb') as f:
    results = pickle.load(f)

print("=" * 50)
print("CLUSTER ANALYSIS")
print("=" * 50)

# 1. Câte samples în fiecare cluster?
cluster_info = results.get('cluster_info', {})
print("\n📊 Cluster sizes:")
for k, v in cluster_info.items():
    print(f"   Cluster {k}: {v}")

# 2. Keywords per cluster
print("\n📊 Keywords per cluster:")
for k, v in results.get('cluster_keywords', {}).items():
    print(f"   Cluster {k}: {len(v)} keywords → {v[:5] if v else 'EMPTY'}")

# 3. GT distribution per cluster
print("\n📊 GT distribution per cluster:")
for k, v in results.get('cluster_gt_distribution', {}).items():
    print(f"   Cluster {k}: {v}")

# 4. K final vs K ground truth
print(f"\n📊 K_final: {results.get('K_final')}")
print(f"📊 K_ground_truth: {results.get('K_ground_truth')}")

# 5. OOD detection stats
ood = results.get('ood_detection', {})
print(f"\n📊 OOD Detection:")
print(f"   Total samples: {ood.get('total_samples')}")
print(f"   Unknown (OOD): {ood.get('unknown_count')}")
print(f"   Known (ID): {ood.get('known_count')}")
