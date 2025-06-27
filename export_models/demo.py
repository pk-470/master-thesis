if __name__ == "__main__":
    import torch
    import torch.optim as optim
    from torch.nn import MSELoss
    from torchinfo import summary

    from spec_mamba.models.audio_mamba import AudioMamba
    from spec_mamba.models.ssast import SSAST

    # Configuration
    spec_size = (80, 129)  # (F, T)
    patch_size = (16, 3)
    channels = 2
    embed_dim = 192
    batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.randn(batch_size, channels, *spec_size, device=device)

    # Initialize the AudioMamba model, e.g. with 'output_type="full"', 'use_pred_head=True'.
    #
    # Option 'use_pred_head=True' initializes an MLP prediction head with the same
    # output dimension as the patch dimension. A different prediction head is applied
    # on the CLS token and the patch embeddings. Set 'use_pred_head=False' to
    # not initialize the prediction head and return the raw embeddings with the
    # 'forward' method instead. Raw embeddings are always accessible with the
    # 'forward_features' method.
    #
    # Input: a batched spectrogram of shape (B, C, F, T).
    #
    # Output: option 'output_type="full"' returns two tensors: (cls, embeddings) and mask.
    # Note that the CLS token is appended to the beginning of the output tensor regardless
    # of its position in the internal forward pass. Possible output types:
    # - "full": (cls, embeddings), mask
    # - "cls" cls, mask
    # - "emb" embeddings, mask
    # - "last" (last embedding), mask
    # - "mean" (average pooling of embeddings), mask
    # - "max" (max pooling of embeddings), mask
    #
    # Use the 'reshape_as_spec' method to reshape the patch embeddings to the original spectrogram shape.
    #
    # Possible CLS positions: "none", "start", "middle", "end", "double" (both start and end).

    model = AudioMamba(
        spec_size=spec_size,
        patch_size=patch_size,
        channels=channels,
        embed_dim=embed_dim,
        depth=5,
        ssm_cfg={"d_state": 24, "d_conv": 4, "expand": 3},  # Inner SSM config
        cls_position="middle",
        output_type="full",
        use_pred_head=True,
        mask_ratio=0.4,
    ).to(device)

    # Similarly:
    # model = SSAST(
    #     spec_size=spec_size,
    #     patch_size=patch_size,
    #     channels=channels,
    #     embed_dim=embed_dim,
    #     depth=5,
    #     num_heads=12,  # Embedding dimension must be divisible by num_heads
    #     cls_position="middle",
    #     output_type="full",
    #     use_pred_head=True,
    #     mask_ratio=0.4,
    # ).to(device)

    model.eval()

    # Print parameters
    print()
    print("Spectrogram parameters:")
    print(f"Spec size (F, T): {model.spec_size}")
    print(f"Channels (C): {model.channels}")
    print(f"Patch size (H, W): {model.patch_size}")
    print(f"Patch dimension (H x W x C): {model.patch_dim}")
    print(f"Number of patches (N = (F / H) x (T / W)): {model.num_patches}")
    print(f"Embeddings dimension (D): {model.embed_dim}")
    print(f"Mask ratio: {model.mask_ratio}")

    # Print model summary
    print()
    print("Model summary:")
    summary(
        model,
        input_data=sample,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
        ],
        depth=6,
        device=device,
    )

    # For inference you can specify different output types in the forward method.
    print()
    print("forward method output shapes:")
    for out_type in ("full", "cls", "emb", "last", "mean", "max"):
        with torch.no_grad():
            output, mask = model(sample, output_type=out_type)
        print(f"Output type '{out_type}': {output.shape=}, {mask.shape=}")

    # Get raw embeddings (before prediction head) with the forward_features method.
    print()
    print("forward_features method output shapes:")
    for out_type in ("full", "cls", "emb", "last", "mean", "max"):
        with torch.no_grad():
            output, mask = model.forward_features(sample, output_type=out_type)
        print(f"Output type '{out_type}': {output.shape=}, {mask.shape=}")

    # ----------------------------------------------------
    # Train for one epoch
    # ----------------------------------------------------

    print()
    print("Training for one epoch:")

    optimizer = optim.AdamW(model.parameters())
    loss_fn = MSELoss()

    model.train()
    optimizer.zero_grad()

    # For training on (masked) reconstruction, reshape the patch embeddings as
    # a spectrogram before applying the loss function.
    logits, mask = model(sample)
    reconstruction, mask = model.reshape_as_spec(logits[:, 1:, :], mask)

    # Alternatively:
    # logits = model(sample, output_type="emb")
    # reconstruction, mask = model.reshape_as_spec(logits, mask)

    loss = loss_fn(reconstruction, sample)
    print(f"Training loss: {loss.item():.4f}")

    loss.backward()
    optimizer.step()

    # ----------------------------------------------------
    # Save weights
    # ----------------------------------------------------

    print()
    print("Saving weights:")

    model.eval()
    torch.save(model.state_dict(), "aum_demo.pt")
    model.load_state_dict(torch.load("aum_demo.pt", weights_only=True))

    print("Weights saved at 'aum_demo.pt'.")
    print()
