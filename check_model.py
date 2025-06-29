import torch
import sys

def check_model(path):
    print(f"Checking model file: {path}")
    try:
        state = torch.load(path, map_location="cpu")
        print(f"Type of loaded data: {type(state)}")
        
        if isinstance(state, dict):
            print(f"Number of keys: {len(state)}")
            print(f"Some key names: {list(state.keys())[:5]}")
            for k in list(state.keys())[:3]:
                v = state[k]
                print(f"Key: {k}, Shape: {v.shape}, Type: {v.dtype}")
        elif hasattr(state, 'state_dict'):
            state_dict = state.state_dict()
            print(f"Number of keys in state_dict: {len(state_dict)}")
            print(f"Some key names: {list(state_dict.keys())[:5]}")
        else:
            print(f"Unexpected type: {type(state)}")
        
        print("Model file appears valid")
        return True
    except Exception as e:
        print(f"Error loading model file: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_model.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    if check_model(model_path):
        sys.exit(0)
    else:
        sys.exit(1)
