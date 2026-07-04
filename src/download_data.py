import moscot.datasets
import os
import scanpy as sc

def main():
    # Define local data directory
    data_dir = "/mnt/d/001find/project_4_Spatiotemporal_Lineage/data"
    os.makedirs(data_dir, exist_ok=True)
    
    dest_path = os.path.join(data_dir, "mosta.h5ad")
    print(f"[Download] Target download path: {dest_path}")
    
    # Trigger download via moscot datasets
    print("[Download] Fetching MOSTA (Mouse Organogenesis) dataset...")
    adata = moscot.datasets.mosta(path=dest_path, force_download=False)
    
    print("\n[Download] Success! Loaded AnnData details:")
    print(adata)
    print("\n[Download] Observations keys (obs):", list(adata.obs.keys()))
    print("[Download] Unstructured keys (uns):", list(adata.uns.keys()))
    if 'spatial' in adata.obsm:
        print("[Download] Spatial coordinates found in obsm['spatial'] of shape:", adata.obsm['spatial'].shape)

if __name__ == "__main__":
    main()
