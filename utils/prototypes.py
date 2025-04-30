import json
import torch
import os
import numpy as np

def generate_random_orthogonal_matrix(feature_dim, num_classes):
    a = np.random.random(size=(feature_dim, num_classes))
    P, _ = np.linalg.qr(a)
    P = torch.tensor(P, dtype=torch.float32)
    assert torch.allclose(P.T @ P, torch.eye(num_classes), atol=1e-7), "Orthogonality issue!"
    return P

def generate_etf_class_prototypes(feature_dim, num_classes, device='cuda', save_dir="./etf_prototypes"):
    """
    Generate or load ETF class prototypes with shape validation, metadata, and dual-format saving.
    File: etf_prototypes_K{num_classes}_D{feature_dim}.pt
    """
    os.makedirs(save_dir, exist_ok=True)
    base_name = f"etf_prototypes_K{num_classes}_D{feature_dim}"
    pt_path = os.path.join(save_dir, base_name + ".pt")
    npy_path = os.path.join(save_dir, base_name + ".npy")
    meta_path = os.path.join(save_dir, base_name + ".json")

    def save_all(prototypes):
        torch.save(prototypes, pt_path)
        np.save(npy_path, prototypes.cpu().numpy())
        with open(meta_path, 'w') as f:
            json.dump({"num_classes": num_classes, "feature_dim": feature_dim}, f)
        print(f"[ETF] Saved to: {pt_path}, {npy_path}, {meta_path}")

    def is_valid_shape(tensor):
        return tensor.shape == (num_classes, feature_dim)

    # -------------------
    # Attempt Load
    # -------------------
    if os.path.exists(pt_path) and os.path.exists(meta_path):
        try:
            M_star = torch.load(pt_path, map_location=device)
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            # Validate shape and metadata
            if not is_valid_shape(M_star):
                print(f"[ETF] Invalid shape: found {M_star.shape}, expected ({num_classes}, {feature_dim}). Regenerating.")
                raise ValueError("Shape mismatch")

            if meta["num_classes"] != num_classes or meta["feature_dim"] != feature_dim:
                print(f"[ETF] Metadata mismatch: {meta}. Regenerating.")
                raise ValueError("Metadata mismatch")

            print(f"[ETF] Loaded existing prototypes from: {pt_path}")
            return M_star.to(device)

        except Exception as e:
            print(f"[ETF] Error loading saved prototype. Regenerating... Reason: {str(e)}")

    # -------------------
    # Generate New
    # -------------------
    print(f"[ETF] Generating new ETF prototypes (K={num_classes}, d={feature_dim}).")
    P = generate_random_orthogonal_matrix(feature_dim, num_classes)  # [D, K]
    I = torch.eye(num_classes)
    ones = torch.ones(num_classes, num_classes)
    M_star = np.sqrt(num_classes / (num_classes - 1)) * P @ (I - (1 / num_classes) * ones)  # [D, K]
    M_star = M_star.T.contiguous().to(device)  # Final shape [K, D]

    save_all(M_star)
    return M_star