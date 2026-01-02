import pandas as pd
import numpy as np

try:
    results_path = 'artifacts/minute_share/inference_test/2025-12-23/run=test_run_share/minutes.parquet'
    df_res = pd.read_parquet(results_path)
    
    features_path = '/home/daniel/projections-data/live/features_minutes_v1/2025-12-23/run=20251224T045508Z/features.parquet'
    # Load just enough to map IDs
    df_feat = pd.read_parquet(features_path, columns=['player_id', 'player_name', 'team_tricode'])
    # Deduplicate features just in case
    df_feat = df_feat.drop_duplicates(subset=['player_id'])
    
    # Merge
    df = df_res.merge(df_feat, on='player_id', how='left')
    
    print("Columns:", df.columns.tolist())
    
    # Check LAL and BOS specifically
    # If team_tricode is missing (NaN), we can't search.
    # We can try to map team_id if tricode fails.
    
    targets = ['LAL', 'BOS', 'OKC', 'MIN']
    
    for team in targets:
        team_slice = df[df['team_tricode'] == team]
        if team_slice.empty:
            continue
            
        print(f"\n=== SPOT CHECK: {team} ===")
        team_slice = team_slice.sort_values('minutes_p50', ascending=False)
        cols = ['player_name', 'starter_flag', 'minutes_p50', 'play_prob']
        # If play_prob missing, skip it
        cols = [c for c in cols if c in team_slice.columns]
        
        print(team_slice[cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
        print(f"Total: {team_slice['minutes_p50'].sum():.1f}")

except Exception as e:
    import traceback
    traceback.print_exc()
