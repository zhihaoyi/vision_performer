import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, random_features, kernel_transformation, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.random_features = random_features
        self.kernel_transformation = kernel_transformation
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)
        
    #####################################################################
    #                      Fast Attention Block                         #    
    #                             Start                                 #
    #####################################################################
    
    '''
Reuse Code From:
https://github.com/google-research/google-research/blob/6c90b3271babc1a3bd69edda761562556b3d8596/performer/fast_attention/tensorflow/fast_attention.py#L28

@inproceedings{performer,
  author    = {Krzysztof Choromanski and
               Valerii Likhosherstov and
               David Dohan and
               Xingyou Song and
               Andreea Gane and
               Tam{\'{a}}s Sarl{\'{o}}s and
               Peter Hawkins and
               Jared Davis and
               Afroz Mohiuddin and
               Lukasz Kaiser and
               David Belanger and
               Lucy Colwell and
               Adrian Weller},
  title     = {Rethinking Attention with Performers},
  booktitle = {International Conference on Learning Representations, {ICLR} 2021},
  year      = {2021},
}
    '''
    
    def relu_kernel_transformation(self, data, projection_matrix, numerical_stabilizer=0.001):
        ratio = 1.0 / tf.math.sqrt(tf.dtypes.cast(projection_matrix.shape[0], tf.float32))
        data_dash = ratio * tf.einsum("blhd,md->blhm", data, projection_matrix)
        return tf.nn.relu(data_dash) + numerical_stabilizer
    
    def softmax_kernel_transformation(self, data, projection_matrix, numerical_stabilizer=0.000001):       
        data_normalizer = 1.0 / (tf.math.sqrt(tf.math.sqrt(tf.dtypes.cast(self.projection_dim, tf.float32))))
        data = data_normalizer * data
        ratio = 1.0 / tf.math.sqrt(tf.dtypes.cast(projection_matrix.shape[0], tf.float32))
        data_dash = tf.einsum("blhd,md->blhm", data, projection_matrix)
        diag_data = tf.math.square(data)
        diag_data = tf.math.reduce_sum(diag_data, axis=tf.keras.backend.ndim(data) - 1)
        diag_data = diag_data / 2.0
        diag_data = tf.expand_dims(diag_data, axis=tf.keras.backend.ndim(data) - 1)
        last_dims_t = (len(data_dash.shape) - 1,)
        attention_dims_t = (len(data_dash.shape) - 3,)
        data_dash = ratio * (tf.math.exp(data_dash - diag_data - tf.math.reduce_max(data_dash, axis=last_dims_t, keepdims=True)) + numerical_stabilizer)
        return data_dash
    
    def numerator(self, qs, ks, vs):
        kvs = tf.einsum("lbhm,lbhd->bhmd", ks, vs)
        return tf.einsum("lbhm,bhmd->lbhd", qs, kvs)
    
    def denominator(self, qs, ks):
        all_ones = tf.ones([ks.shape[0]])
        ks_sum = tf.einsum("lbhm,l->bhm", ks, all_ones)
        return tf.einsum("lbhm,bhm->lbh", qs, ks_sum)
    
    def fast_attention(self, q, k, v, kernel_transformation, projection_matrix):
        # qkv [None, None, 8, 64] [B,L,H,D]
        if kernel_transformation == 'softmax':
            q_prime = self.softmax_kernel_transformation(q, projection_matrix) # [B,L,H,M]
            k_prime = self.softmax_kernel_transformation(k, projection_matrix) # [B,L,H,M]
        elif kernel_transformation == 'relu':
            q_prime = self.relu_kernel_transformation(q, projection_matrix) # [B,L,H,M]
            k_prime = self.relu_kernel_transformation(k, projection_matrix) # [B,L,H,M]
        q_prime = tf.transpose(q_prime, [1, 0, 2, 3]) # [L,B,H,M]
        k_prime = tf.transpose(k_prime, [1, 0, 2, 3]) # [L,B,H,M]
        v = tf.transpose(v, [1, 0, 2, 3]) # [L,B,H,D]
        
        av_attention = self.numerator(q_prime, k_prime, v)
        attention_normalizer = self.denominator(q_prime, k_prime)
        av_attention = tf.transpose(av_attention, [1, 0, 2, 3])
        attention_normalizer = tf.transpose(attention_normalizer, [1, 0, 2])
        attention_normalizer = tf.expand_dims(attention_normalizer,
                                              len(attention_normalizer.shape))            
        return av_attention / attention_normalizer
    
    def iid_gaussian(self, m, d):
        return np.random.normal(size=(m, d))
    
    def create_projection_matrix(self, m, d, seed=0, scaling=0, struct_mode=False):
        nb_full_blocks = int(m / d)
        block_list = []
        current_seed = seed
        for _ in range(nb_full_blocks):
            if struct_mode:
                q = create_products_of_givens_rotations(d, seed)
            else:
                unstructured_block = tf.random.normal((d, d), seed=current_seed)
                q, _ = tf.linalg.qr(unstructured_block)
                q = tf.transpose(q)
            block_list.append(q)
            current_seed += 1
        remaining_rows = m - nb_full_blocks * d
        if remaining_rows > 0:
            if struct_mode:
                q = create_products_of_givens_rotations(d, seed)
            else:
                unstructured_block = tf.random.normal((d, d), seed=current_seed)
                q, _ = tf.linalg.qr(unstructured_block)
                q = tf.transpose(q)
            block_list.append(q[0:remaining_rows])
        final_matrix = tf.experimental.numpy.vstack(block_list)
        current_seed += 1

        if scaling == 0:
            multiplier = tf.norm(tf.random.normal((m, d), seed=current_seed), axis=1)
        elif scaling == 1:
            multiplier = tf.math.sqrt(float(d)) * tf.ones((m))
        else:
            raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

        return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)
    
    #####################################################################
    #                      Fast Attention Block                         #    
    #                             End                                   #
    #####################################################################

    def separate_heads(self, x, batch_size): # [None, 65, 512]
        # batch_size = None
        # self.num_heads = H
        # self.projection_dim = embed_dim // num_heads
        transformer_length = x.shape[1] # 65
        x = tf.reshape(
            x, (batch_size, transformer_length, self.num_heads, self.projection_dim) # -1
        ) # [None, None, 8, 64]
        return x

    def call(self, inputs): # [None, 65, 512] [None, patch_num + class, d_model]
        batch_size = tf.shape(inputs)[0] # None
        query = self.query_dense(inputs) # [None, 65, 512]
        key = self.key_dense(inputs)
        value = self.value_dense(inputs) 
        query = self.separate_heads(query, batch_size) # [None, 8, None, 64]
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
    #####################################################################
    #                             ViP Call                              #    
    #                              Start                                #
    #####################################################################
    
        orth_feats = self.create_projection_matrix(self.random_features, self.projection_dim)
        attention = self.fast_attention(query, key, value, self.kernel_transformation, orth_feats)
        
    #####################################################################
    #                             ViP Call                              #    
    #                               End                                 #
    #####################################################################
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) # [None, None, 8, 64]
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention) # [None, None, 512]
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, random_features, kernel_transformation, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, random_features, kernel_transformation, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                Dense(mlp_dim, activation=tfa.activations.gelu),
                Dropout(dropout),
                Dense(embed_dim),
                Dropout(dropout),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1


class VisionPerformer(tf.keras.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        random_features,
        kernel_transformation,
        channels=3,
        dropout=0.1,
    ):
        super(VisionPerformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.random_features = random_features

        self.rescale = Rescaling(1.0 / 255)
        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, num_patches + 1, d_model)
        )
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))
        self.patch_proj = Dense(d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, random_features, kernel_transformation, dropout)
            for _ in range(num_layers)
        ]
        self.mlp_head = tf.keras.Sequential(
            [
                LayerNormalization(epsilon=1e-6),
                Dense(mlp_dim, activation=tfa.activations.gelu),
                Dropout(dropout),
                Dense(num_classes),
            ]
        )

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        x = self.rescale(x)
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)

        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        # First (class token) is used for classification
        x = self.mlp_head(x[:, 0])
        return x
