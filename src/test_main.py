#test_main()

# ============================= main ==================================
from data_pipeline import DataIngestion  # adjust name/path if needed
from models_pipline import ModelsPipeline
from custom_messages import CustomMessages


def main():

    
    cm = CustomMessages()
    cm.rtm("Running data pipeline...")
    dp = DataIngestion(seed=42, n_bins=5, stratify=True, use_quantile_bins=False)
    X_train, y_train, X_val, y_val, X_test, y_test = dp.run()

    cm.rtm("Running models pipeline...")
    mp = ModelsPipeline(drop_cols=["Campaign Call Duration_bin"], random_state=42)
    mp.fit(X_train, y_train)

    val_met = mp.evaluate(X_val, y_val, split_name="val")
    test_met = mp.evaluate(X_test, y_test, split_name="test")

    cm.rtm("Validation metrics:" + val_met.to_string(index=False))
    cm.rtm("Test metrics:" + test_met.to_string(index=False))

    best = test_met.sort_values("f1_macro", ascending=False).iloc[0]["model"]
    rep, cmx = mp.detailed_report(X_test, y_test, best)
    cm.rtm(f"=== Best model on test: {best} === {rep}")
    cm.rtm("Confusion matrix: " + str(cmx))


if __name__ == "__main__":
    main()