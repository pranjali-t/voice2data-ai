"""
Comprehensive automated test suite for Voice2Data AI.
Validates the entire pipeline (Intent Extraction, Validation, Pandas Engine, Visualization Correctness)
against all 10 capstone test commands, semantic visualization tests, and security edge cases.
"""
import sys
import os
import io
import pandas as pd

# Ensure UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from engine.gemini_engine import GeminiEngine, rule_based_fallback_intent
from engine.validator import validate_intent
from engine.pandas_engine import execute_intent
from utils.data_loader import load_default_dataset, get_dataset_quality_profile
from utils.analytics import compute_kpis
from utils.visualization import create_auto_chart

def run_all_tests():
    print("=" * 60)
    print("RUNNING VOICE2DATA AI CAPSTONE TEST SUITE")
    print("=" * 60)

    # 1. Test Dataset Loading
    df = load_default_dataset()
    assert df is not None and not df.empty, "Failed to load default dataset"
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"
    print(f"[PASS] Default Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Test Quality Profile & Analytics
    profile = get_dataset_quality_profile(df)
    assert profile["total_rows"] == 1000
    assert profile["total_cols"] == 12
    kpis = compute_kpis(df, df, execution_count=0)
    assert kpis["rows"] == 1000
    print("[PASS] Quality Profiler & Analytics KPI computation verified.")

    # 3. Required 10 Test Commands
    test_cases = [
        {
            "id": 1,
            "cmd": "Show top 10 customers by revenue.",
            "expect_valid": True,
            "expect_op": "top_n",
            "verify": lambda res: len(res["result_df"]) <= 10 and ("Revenue" in res["result_df"].columns or "Customer" in res["result_df"].columns)
        },
        {
            "id": 2,
            "cmd": "Filter Q3 transactions.",
            "expect_valid": True,
            "expect_op": "filter",
            "verify": lambda res: len(res["result_df"]) > 0 and (res["result_df"]["Quarter"] == "Q3").all()
        },
        {
            "id": 3,
            "cmd": "Sort products by units sold.",
            "expect_valid": True,
            "expect_op": "sort",
            "verify": lambda res: len(res["result_df"]) == 1000
        },
        {
            "id": 4,
            "cmd": "Group revenue by region.",
            "expect_valid": True,
            "expect_op": "groupby",
            "verify": lambda res: "Region" in res["result_df"].columns and len(res["result_df"]) <= 10
        },
        {
            "id": 5,
            "cmd": "Calculate average revenue by category.",
            "expect_valid": True,
            "expect_op": "groupby",
            "verify": lambda res: "Category" in res["result_df"].columns and len(res["result_df"]) <= 10
        },
        {
            "id": 6,
            "cmd": "Show electronics sales above 50000.",
            "expect_valid": True,
            "expect_op": "filter",
            "verify": lambda res: len(res["result_df"]) > 0 and (res["result_df"]["Revenue"] > 50000).all() and (res["result_df"]["Category"] == "Electronics").all()
        },
        {
            "id": 7,
            "cmd": "Select Customer, Revenue and Region columns.",
            "expect_valid": True,
            "expect_op": "select_columns",
            "verify": lambda res: list(res["result_df"].columns) == ["Customer", "Revenue", "Region"]
        },
        {
            "id": 8,
            "cmd": "Show bottom 5 transactions by revenue.",
            "expect_valid": True,
            "expect_op": "bottom_n",
            "verify": lambda res: len(res["result_df"]) == 5
        },
        {
            "id": 9,
            "cmd": "Invalid column command.",
            "expect_valid": False,
            "expect_op": None,
            "verify": None
        },
        {
            "id": 10,
            "cmd": "Unsupported command.",
            "expect_valid": False,
            "expect_op": None,
            "verify": None
        }
    ]

    engine = GeminiEngine()
    passed_count = 0

    print("\n--- TESTING 10 CORE NATURAL LANGUAGE COMMANDS ---")
    for tc in test_cases:
        cmd_text = tc["cmd"]
        intent, _ = engine.extract_intent(cmd_text, df)
        is_valid, validated_intent, err = validate_intent(intent, df)

        if tc["expect_valid"]:
            if not is_valid:
                print(f"[FAIL] Test {tc['id']} FAILED: Expected valid intent for '{cmd_text}', got error: {err}")
                continue
            
            exec_res = execute_intent(df, validated_intent)
            if not exec_res["success"]:
                print(f"[FAIL] Test {tc['id']} FAILED: Pandas execution error: {exec_res['error']}")
                continue

            if tc["verify"] and not tc["verify"](exec_res):
                print(f"[FAIL] Test {tc['id']} FAILED: Verification check failed for '{cmd_text}'")
                continue

            # Check visualization generator
            fig = create_auto_chart(exec_res["result_df"], title_hint=cmd_text, intent=validated_intent)
            
            print(f"[PASS] Test {tc['id']} PASSED: '{cmd_text}' -> Op: {validated_intent.get('operation')} | Rows: {exec_res['rows_after']} | Latency: {exec_res['execution_time_ms']}ms")
            passed_count += 1
        else:
            # Negative test: should safely be invalid or unsupported
            if not is_valid:
                print(f"[PASS] Test {tc['id']} PASSED (Safely Rejected): '{cmd_text}' -> Reason: {err or intent.get('reason')}")
                passed_count += 1
            else:
                print(f"[FAIL] Test {tc['id']} FAILED: Expected rejection for '{cmd_text}', but passed validation.")

    # 4. Semantic Visualization Correctness Regression Tests
    print("\n--- TESTING SEMANTIC VISUALIZATION CORRECTNESS ---")
    vis_passed = 0
    vis_cases = [
        {
            "name": "Bottom 5 transactions by revenue (must use Revenue, not Units)",
            "cmd": "Show bottom 5 transactions by revenue.",
            "target_col": "Revenue",
            "forbidden_col": "Units"
        },
        {
            "name": "Top 10 customers by revenue (must use Revenue, not Units)",
            "cmd": "Show top 10 customers by revenue.",
            "target_col": "Revenue",
            "forbidden_col": "Units"
        },
        {
            "name": "Calculate average revenue by category (must use Revenue, not Units)",
            "cmd": "Calculate average revenue by category.",
            "target_col": "Revenue",
            "forbidden_col": "Units"
        }
    ]

    def _extract_numeric_vals(raw_list):
        out = []
        if raw_list is None:
            return out
        for v in raw_list:
            try:
                out.append(float(v))
            except (ValueError, TypeError):
                pass
        return out

    for v_idx, vc in enumerate(vis_cases, 1):
        intent, _ = engine.extract_intent(vc["cmd"], df)
        _, val_intent, _ = validate_intent(intent, df)
        exec_res = execute_intent(df, val_intent)
        res_df = exec_res["result_df"]
        fig = create_auto_chart(res_df, title_hint=vc["cmd"], intent=val_intent)

        assert fig is not None, f"Visualization failed to generate for {vc['name']}"
        
        # Check trace values
        trace = fig.data[0]
        y_vals = _extract_numeric_vals(getattr(trace, 'y', None))
        x_vals = _extract_numeric_vals(getattr(trace, 'x', None))
        val_vals = _extract_numeric_vals(getattr(trace, 'values', None))

        chart_vals = y_vals if y_vals else (x_vals if x_vals else val_vals)

        if vc["target_col"] in res_df.columns:
            expected_vals = [float(v) for v in res_df[vc["target_col"]]]
            assert set(chart_vals) == set(expected_vals), f"Chart values mismatch for {vc['name']}"

        if vc["forbidden_col"] in res_df.columns and vc["target_col"] != vc["forbidden_col"]:
            units_vals = [float(v) for v in res_df[vc["forbidden_col"]]]
            if set(units_vals) != set(expected_vals):
                assert set(chart_vals) != set(units_vals), f"Chart incorrectly used {vc['forbidden_col']} for {vc['name']}"

        print(f"[VIS-PASS] Vis Check #{v_idx} PASSED: {vc['name']}")
        vis_passed += 1

    # 5. Test Direct Security Code Injection Rejection
    print("\n--- TESTING SECURITY BOUNDARIES (NO CODE EXECUTION) ---")
    malicious_intents = [
        {"operation": "execute_python", "code": "import os; os.system('calc')"},
        {"operation": "filter", "conditions": [{"column": "eval('Revenue')", "operator": "==", "value": 100}]},
        {"operation": "top_n", "column": "NonExistentColumn", "n": -50},
        {"operation": "groupby", "group_by": ["FakeColumn"], "aggregations": [{"column": "Revenue", "agg_func": "exec"}]}
    ]

    sec_passed = 0
    for i, m_intent in enumerate(malicious_intents, 1):
        is_valid, _, val_err = validate_intent(m_intent, df)
        if not is_valid:
            print(f"[SHIELD] Security Check #{i} PASSED (Safely blocked): {val_err}")
            sec_passed += 1
        else:
            print(f"[ALERT] Security Check #{i} FAILED: Malicious payload was not blocked!")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed_count}/{len(test_cases)} Commands Passed | {vis_passed}/{len(vis_cases)} Visualizations Passed | {sec_passed}/{len(malicious_intents)} Security Checks Passed")
    print("=" * 60)

    if passed_count == len(test_cases) and vis_passed == len(vis_cases) and sec_passed == len(malicious_intents):
        print("ALL TESTS PASSED SUCCESSFULLY!")
        return True
    else:
        print("SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
