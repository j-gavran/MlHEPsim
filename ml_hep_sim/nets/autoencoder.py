from torch import nn


class Encoder(nn.Module):
    def __init__(self, seq_encoder):
        super().__init__()
        self.encoder = seq_encoder

    def forward(self, x):
        encoder = self.encoder[:-2]
        mu_layer, logvar_layer = self.encoder[-2], self.encoder[-1]

        x = encoder(x)
        h21, h22 = mu_layer(x), logvar_layer(x)

        return h21, h22


class Decoder(nn.Module):
    def __init__(self, seq_decoder):
        super().__init__()
        self.decoder = seq_decoder

    def forward(self, z):
        decoder = self.decoder
        return decoder(z)


class AEBuilder:
    def __init__(self, input_dim, hidden_layers, latent_dim, activation, output_activation="Identity"):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim
        self.activation = activation
        self.output_activation = output_activation

        self.encoder_layers, self.decoder_layers = [], []

    def make_encoder_layers(self):
        self.encoder_layers.append([self.input_dim, self.hidden_layers[0]])

        for i in range(len(self.hidden_layers[1:])):
            self.encoder_layers.append([self.hidden_layers[i], self.hidden_layers[i + 1]])

        self.encoder_layers.append([self.encoder_layers[-1][1], self.latent_dim])
        self.encoder_layers.append([self.encoder_layers[-2][1], self.latent_dim])

    def make_decoder_layers(self):
        reversed_hideen_layers = self.hidden_layers[::-1]
        self.decoder_layers.append([self.latent_dim, reversed_hideen_layers[0]])

        for i in range(len(reversed_hideen_layers[1:])):
            self.decoder_layers.append([reversed_hideen_layers[i], reversed_hideen_layers[i + 1]])

        self.decoder_layers.append([self.decoder_layers[-1][1], self.input_dim])

    def build_encoder(self, layers):
        self.make_encoder_layers()
        activation_function = getattr(nn, self.activation)

        encoder = []
        for i, layer_dim in enumerate(layers):
            encoder.append(nn.Linear(layer_dim[0], layer_dim[1]))
            if i < len(layers) - 2:
                encoder.append(activation_function())

        encoder = nn.Sequential(*encoder)
        return encoder

    def build_decoder(self, layers):
        self.make_decoder_layers()
        activation_function = getattr(nn, self.activation)

        decoder = []
        for i, layer_dim in enumerate(layers):
            decoder.append(nn.Linear(layer_dim[0], layer_dim[1]))
            if i != len(layers) - 1:
                decoder.append(activation_function())

        output_activation_function = getattr(nn, self.output_activation)
        decoder.append(output_activation_function())

        decoder = nn.Sequential(*decoder)
        return decoder

    def build_autoencoder(self):
        encoder = self.build_encoder(self.encoder_layers)
        decoder = self.build_decoder(self.decoder_layers)
        self.encoder, self.decoder = Encoder(encoder), Decoder(decoder)

        return self.encoder, self.decoder


class Autoencoder:
    def __init__(self, input_dim, hidden_layers, latent_dim, activation, output_activation="Identity"):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim
        self.activation = activation
        self.output_activation = output_activation

        ae_builder = AEBuilder(input_dim, hidden_layers, latent_dim, activation, output_activation)
        self.encoder, self.decoder = ae_builder.build_autoencoder()


if __name__ == "__main__":
    import torch
    from ml_hep_sim.ml_utils import netron_model

    ae = Autoencoder(
        input_dim=18,
        hidden_layers=[128, 128, 128, 128, 128, 128, 128],
        latent_dim=14,
        activation="ELU",
        output_activation="Tanh",
    )

    enc, dec = ae.encoder, ae.decoder

    dummy_enc, dummy_dec = torch.randn((100, 18)), torch.randn((100, 14))

    netron_model(enc, dummy_enc)
    netron_model(dec, dummy_dec)
