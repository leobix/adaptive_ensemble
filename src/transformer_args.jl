function add_transformer_args!(s)
    @add_arg_table! s begin
        "--fedformer"
            help = "Enable FEDFormer baseline"
            action = :store_true

        "--fedformer_seq_len"
            help = "FEDFormer encoder sequence length"
            arg_type = Int
            default = 24

        "--fedformer_label_len"
            help = "FEDFormer decoder label length"
            arg_type = Int
            default = 12

        "--fedformer_pred_len"
            help = "FEDFormer prediction length"
            arg_type = Int
            default = 1

        "--fedformer_d_model"
            help = "FEDFormer model dimension"
            arg_type = Int
            default = 64

        "--fedformer_n_heads"
            help = "FEDFormer attention heads"
            arg_type = Int
            default = 4

        "--fedformer_e_layers"
            help = "FEDFormer encoder layers"
            arg_type = Int
            default = 2

        "--fedformer_d_layers"
            help = "FEDFormer decoder layers"
            arg_type = Int
            default = 1

        "--fedformer_d_ff"
            help = "FEDFormer feed-forward dimension"
            arg_type = Int
            default = 128

        "--fedformer_dropout"
            help = "FEDFormer dropout"
            arg_type = Float64
            default = 0.1

        "--fedformer_activation"
            help = "FEDFormer activation (gelu|relu|linear)"
            arg_type = String
            default = "gelu"

        "--fedformer_moving_avg"
            help = "FEDFormer moving average window"
            arg_type = Int
            default = 5

        "--fedformer_freq_mode"
            help = "FEDFormer frequency mode (fourier|wavelet)"
            arg_type = String
            default = "fourier"

        "--fedformer_modes"
            help = "FEDFormer number of frequency modes / wavelet levels"
            arg_type = Int
            default = 16

        "--fedformer_epochs"
            help = "FEDFormer training epochs"
            arg_type = Int
            default = 5

        "--fedformer_batch_size"
            help = "FEDFormer batch size"
            arg_type = Int
            default = 16

        "--fedformer_lr"
            help = "FEDFormer learning rate"
            arg_type = Float64
            default = 0.001

        "--fedformer_save_preds"
            help = "Save FEDFormer standardized predictions to results_beta/"
            action = :store_true

        "--informer"
            help = "Enable Informer baseline"
            action = :store_true

        "--informer_seq_len"
            help = "Informer encoder sequence length"
            arg_type = Int
            default = 24

        "--informer_label_len"
            help = "Informer decoder label length"
            arg_type = Int
            default = 12

        "--informer_pred_len"
            help = "Informer prediction length"
            arg_type = Int
            default = 1

        "--informer_d_model"
            help = "Informer model dimension"
            arg_type = Int
            default = 64

        "--informer_n_heads"
            help = "Informer attention heads"
            arg_type = Int
            default = 4

        "--informer_e_layers"
            help = "Informer encoder layers"
            arg_type = Int
            default = 2

        "--informer_d_layers"
            help = "Informer decoder layers"
            arg_type = Int
            default = 1

        "--informer_d_ff"
            help = "Informer feed-forward dimension"
            arg_type = Int
            default = 128

        "--informer_dropout"
            help = "Informer dropout"
            arg_type = Float64
            default = 0.1

        "--informer_activation"
            help = "Informer activation (gelu|relu|linear)"
            arg_type = String
            default = "gelu"

        "--informer_factor"
            help = "Informer ProbSparse top-k factor"
            arg_type = Int
            default = 5

        "--informer_distil"
            help = "Enable Informer encoder distilling between encoder layers"
            action = :store_true

        "--informer_epochs"
            help = "Informer training epochs"
            arg_type = Int
            default = 5

        "--informer_batch_size"
            help = "Informer batch size"
            arg_type = Int
            default = 16

        "--informer_lr"
            help = "Informer learning rate"
            arg_type = Float64
            default = 0.001

        "--informer_save_preds"
            help = "Save Informer standardized predictions to results_beta/"
            action = :store_true
    end
end
