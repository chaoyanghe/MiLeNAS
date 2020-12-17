from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

# CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" V2_r2_e50
V2_r2_e50 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

# CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" V2_r2_e100
V2_r2_e100 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2),
            ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

# CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" V2_r3_e50
V2_r3_e50 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

# CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" V2_r3_e100
V2_r3_e100 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1),
            ('dil_conv_5x5', 2), ('skip_connect', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

# CUDA_VISIBLE_DEVICES=4 sh run_eval_cnn.sh "4" V2_r4_e50
V2_r4_e50 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

# CUDA_VISIBLE_DEVICES=5 sh run_eval_cnn.sh "5" V2_r4_e100
V2_r4_e100 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2),
            ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

# CUDA_VISIBLE_DEVICES=6 sh run_eval_cnn.sh "6" V2_r5_e50
V2_r5_e50 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

# CUDA_VISIBLE_DEVICES=7 sh run_eval_cnn.sh "7" V2_r5_e100
V2_r5_e100 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

v2_r2_e200 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0),
            ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

v2_r3_e200 = Genotype(
    normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 0),
            ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 1),
            ('dil_conv_5x5', 3), ('skip_connect', 3), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

v2_r4_e200 = Genotype(
    normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2),
            ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

v2_r5_e200 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

l1_r0_e200 = Genotype(
    normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2),
            ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))

l05_r2_e200 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0),
            ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1),
            ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

# ----------2nd round searched architecture (CNN)----------

r112_e50_48_n6_r2 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))

r116_e100_92_n5_r1 = Genotype(
    normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

r116_e100_69_n5_r0 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

# ----------after in-place bug fixed-------
r100_e50_48_n7r1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 0)], reduce_concat=range(2, 6))

r101_e50_48_n7r3 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('dil_conv_5x5', 3), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

r102_e50_45_n8r1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0),
            ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

r103_e50_45_n7r2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

# -----e50--------------------------------
r116_e100_95_n3r0 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

r117_e100_98_n3r0 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0),
            ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))

r118_e100_92_n4r1 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

r119_e100_86_n4r0 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2),
            ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

################################################################
# 1.55M
r104_e200_185_n1r0_91_116 = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0),
            ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

# 1.91
r104_e200_197_n2r0_91_188 = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 0),
            ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

r104_e200_131_n3r0_90_528 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

r104_e200_120_n4r0_90_32 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

r104_e200_106_n5r0_89_776 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

r104_e200_50_n6r0_87_056 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

r104_e200_43_n7r0_85_824 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

r306_2nd_e50_49_n7r8 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

r803_2nd_e50_46_n7r1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2),
            ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

#### e80
r904_e80_75_n4_r0 = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

#### e200
r902_e200_192_n2_r1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 3),
            ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))

r902_e200_198_n2_r2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 3),
            ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))

r902_e200_155_n3_r1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))

r902_e200_136_n4_r1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

r902_e200_104_n4_r2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

r902_e200_62_n5_r0 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

r902_e200_64_n5_r1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

r902_e200_42_n6_r0 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

r2001_e50_45_n8_r3 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

GDAS_MIXED_LEVEL1 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1),
                                     ('skip_connect', 0), ('max_pool_3x3', 1),
                                     ('skip_connect', 1),
                                     ('skip_connect', 0), ('skip_connect', 1),
                                     ('sep_conv_5x5', 0)], normal_concat=range(2, 6),
                             reduce=[('sep_conv_5x5', 1),
                                     ('sep_conv_5x5', 0),
                                     ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
                                     ('sep_conv_3x3', 1), ('sep_conv_5x5', 2),
                                     ('sep_conv_5x5', 0),
                                     ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

GDAS_MIXED_LEVEL2 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1),
                                     ('sep_conv_3x3', 0), ('dil_conv_5x5', 2),
                                     ('sep_conv_5x5', 0), ('sep_conv_5x5', 3),
                                     ('sep_conv_3x3', 4),
                                     ('dil_conv_3x3', 2)], normal_concat=range(2, 6),
                             reduce=[('avg_pool_3x3', 1), ('skip_connect', 0),
                                     ('skip_connect', 2), ('sep_conv_5x5', 0),
                                     ('sep_conv_3x3', 3), ('sep_conv_3x3', 0),
                                     ('max_pool_3x3', 0),
                                     ('skip_connect', 4)], reduce_concat=range(2, 6))
