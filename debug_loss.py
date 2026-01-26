import inspect
import sys
import os

# Add the current directory to sys.path so we can import 'models'
sys.path.append(os.getcwd())

try:
    import models.losses as losses
    print(f"losses file: {losses.__file__}")
    
    fn = losses.softmax_cross_entropy
    print(f"softmax_cross_entropy: {fn}")
    print(f"Signature: {inspect.signature(fn)}")
    
    # Check if globals() works as expected
    loss_type = "softmax_cross_entropy"
    resolved_fn = losses.globals().get(loss_type) # globals() is built-in, but inside the module scope we need to access that module's globals.
    # Actually, one cannot access a module's globals from outside easily like this unless we use __dict__.
    resolved_fn_dict = losses.__dict__.get(loss_type)
    print(f"Resolved from dict: {resolved_fn_dict}")
    if resolved_fn_dict:
         print(f"Resolved Signature: {inspect.signature(resolved_fn_dict)}")

except Exception as e:
    print(f"Error: {e}")
