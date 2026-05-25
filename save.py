import os
import tempfile
import mlflow
import mlflow.sklearn
import skops.io as sio

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def export_and_log_best():
    state_file = '.training_state.skops'
    
    if not os.path.exists(state_file):
        print(f"Error: Cache file '{state_file}' not found. Run 'train.py' first.")
        return

    print("Loading temporary training execution cache...")
    untrusted_types = sio.get_untrusted_types(file=state_file)
    state = sio.load(state_file, trusted=untrusted_types)
    
    best_model = state['best_model']
    scaler = state['scaler']
    le = state['encoder']
    class_names = state['classes']
    best_name = state['best_name']
    best_run_id = state['best_run_id']

    print(f"Resuming MLflow Run Session Context: {best_run_id} ({best_name})")
    
    with mlflow.start_run(run_id=best_run_id):
        # Updated artifact_path wrapper parameter implementation to modern 'name' identifier syntax 
        mlflow.sklearn.log_model(best_model, name='model')
        
        save_obj = {'model': best_model, 'scaler': scaler, 'encoder': le, 'classes': class_names}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_skops_path = os.path.join(tmpdir, 'best_model.skops')
            sio.dump(save_obj, temp_skops_path)
            mlflow.log_artifact(local_path=temp_skops_path)
            print("Successfully uploaded secure skops artifact directly to MLflow server.")

        sio.dump(save_obj, 'best_model.skops')
        print("Saved local application serving asset: 'best_model.skops'")
        
    os.remove(state_file)

if __name__ == "__main__":
    export_and_log_best()