//! Unchained Hash Table for Join Processing
//!
//! A high-performance, cache-friendly hash table optimized for equi-join
//! processing on modern multicore CPUs. Based on:
//!
//! "Simple, Efficient, and Robust Hash Tables for Join Processing"
//! (Birler et al., DaMoN '24)
//!
//! # Architecture
//!
//! ```text
//!  Directory (Vec<DirectoryEntry>)     Tuple Storage (Vec<u8>)
//! ┌────────────────────────┐          ┌──────────────────────────┐
//! │ sentinel: off=0  bl=0  │─────────►│                          │
//! ├────────────────────────┤          │ tuples with hash prefix 0│
//! │ slot 0:   off=32 bl=.. │─────────►├──────────────────────────┤
//! │ slot 1:   off=64 bl=.. │─────────►│ tuples with hash prefix 1│
//! │ ...                    │          │ ...                      │
//! └────────────────────────┘          └──────────────────────────┘
//!
//! Each directory entry (u64):
//!   bits [63:16] = byte offset into tuple storage (end of range)
//!   bits [15:0]  = 16-bit Bloom filter
//! ```
use std::mem::size_of;
use std::thread;

// ===========================================================================
// Bloom filter
// ===========================================================================

#[repr(C, align(4096))]
struct BloomTable {
    data: [u16; 2048],
}

static BLOOM_TAGS: BloomTable = BloomTable {
    data: [
        // The 1820 distinct masks (C(16,4) patterns, each with popcount 4)
        15, 23, 27, 29, 30, 39, 43, 45, 46, 51, 53, 54, 57, 58, 60, 71, 75, 77, 78, 83, 85, 86, 89,
        90, 92, 99, 101, 102, 105, 106, 108, 113, 114, 116, 120, 135, 139, 141, 142, 147, 149, 150,
        153, 154, 156, 163, 165, 166, 169, 170, 172, 177, 178, 180, 184, 195, 197, 198, 201, 202,
        204, 209, 210, 212, 216, 225, 226, 228, 232, 240, 263, 267, 269, 270, 275, 277, 278, 281,
        282, 284, 291, 293, 294, 297, 298, 300, 305, 306, 308, 312, 323, 325, 326, 329, 330, 332,
        337, 338, 340, 344, 353, 354, 356, 360, 368, 387, 389, 390, 393, 394, 396, 401, 402, 404,
        408, 417, 418, 420, 424, 432, 449, 450, 452, 456, 464, 480, 519, 523, 525, 526, 531, 533,
        534, 537, 538, 540, 547, 549, 550, 553, 554, 556, 561, 562, 564, 568, 579, 581, 582, 585,
        586, 588, 593, 594, 596, 600, 609, 610, 612, 616, 624, 643, 645, 646, 649, 650, 652, 657,
        658, 660, 664, 673, 674, 676, 680, 688, 705, 706, 708, 712, 720, 736, 771, 773, 774, 777,
        778, 780, 785, 786, 788, 792, 801, 802, 804, 808, 816, 833, 834, 836, 840, 848, 864, 897,
        898, 900, 904, 912, 928, 960, 1031, 1035, 1037, 1038, 1043, 1045, 1046, 1049, 1050, 1052,
        1059, 1061, 1062, 1065, 1066, 1068, 1073, 1074, 1076, 1080, 1091, 1093, 1094, 1097, 1098,
        1100, 1105, 1106, 1108, 1112, 1121, 1122, 1124, 1128, 1136, 1155, 1157, 1158, 1161, 1162,
        1164, 1169, 1170, 1172, 1176, 1185, 1186, 1188, 1192, 1200, 1217, 1218, 1220, 1224, 1232,
        1248, 1283, 1285, 1286, 1289, 1290, 1292, 1297, 1298, 1300, 1304, 1313, 1314, 1316, 1320,
        1328, 1345, 1346, 1348, 1352, 1360, 1376, 1409, 1410, 1412, 1416, 1424, 1440, 1472, 1539,
        1541, 1542, 1545, 1546, 1548, 1553, 1554, 1556, 1560, 1569, 1570, 1572, 1576, 1584, 1601,
        1602, 1604, 1608, 1616, 1632, 1665, 1666, 1668, 1672, 1680, 1696, 1728, 1793, 1794, 1796,
        1800, 1808, 1824, 1856, 1920, 2055, 2059, 2061, 2062, 2067, 2069, 2070, 2073, 2074, 2076,
        2083, 2085, 2086, 2089, 2090, 2092, 2097, 2098, 2100, 2104, 2115, 2117, 2118, 2121, 2122,
        2124, 2129, 2130, 2132, 2136, 2145, 2146, 2148, 2152, 2160, 2179, 2181, 2182, 2185, 2186,
        2188, 2193, 2194, 2196, 2200, 2209, 2210, 2212, 2216, 2224, 2241, 2242, 2244, 2248, 2256,
        2272, 2307, 2309, 2310, 2313, 2314, 2316, 2321, 2322, 2324, 2328, 2337, 2338, 2340, 2344,
        2352, 2369, 2370, 2372, 2376, 2384, 2400, 2433, 2434, 2436, 2440, 2448, 2464, 2496, 2563,
        2565, 2566, 2569, 2570, 2572, 2577, 2578, 2580, 2584, 2593, 2594, 2596, 2600, 2608, 2625,
        2626, 2628, 2632, 2640, 2656, 2689, 2690, 2692, 2696, 2704, 2720, 2752, 2817, 2818, 2820,
        2824, 2832, 2848, 2880, 2944, 3075, 3077, 3078, 3081, 3082, 3084, 3089, 3090, 3092, 3096,
        3105, 3106, 3108, 3112, 3120, 3137, 3138, 3140, 3144, 3152, 3168, 3201, 3202, 3204, 3208,
        3216, 3232, 3264, 3329, 3330, 3332, 3336, 3344, 3360, 3392, 3456, 3585, 3586, 3588, 3592,
        3600, 3616, 3648, 3712, 3840, 4103, 4107, 4109, 4110, 4115, 4117, 4118, 4121, 4122, 4124,
        4131, 4133, 4134, 4137, 4138, 4140, 4145, 4146, 4148, 4152, 4163, 4165, 4166, 4169, 4170,
        4172, 4177, 4178, 4180, 4184, 4193, 4194, 4196, 4200, 4208, 4227, 4229, 4230, 4233, 4234,
        4236, 4241, 4242, 4244, 4248, 4257, 4258, 4260, 4264, 4272, 4289, 4290, 4292, 4296, 4304,
        4320, 4355, 4357, 4358, 4361, 4362, 4364, 4369, 4370, 4372, 4376, 4385, 4386, 4388, 4392,
        4400, 4417, 4418, 4420, 4424, 4432, 4448, 4481, 4482, 4484, 4488, 4496, 4512, 4544, 4611,
        4613, 4614, 4617, 4618, 4620, 4625, 4626, 4628, 4632, 4641, 4642, 4644, 4648, 4656, 4673,
        4674, 4676, 4680, 4688, 4704, 4737, 4738, 4740, 4744, 4752, 4768, 4800, 4865, 4866, 4868,
        4872, 4880, 4896, 4928, 4992, 5123, 5125, 5126, 5129, 5130, 5132, 5137, 5138, 5140, 5144,
        5153, 5154, 5156, 5160, 5168, 5185, 5186, 5188, 5192, 5200, 5216, 5249, 5250, 5252, 5256,
        5264, 5280, 5312, 5377, 5378, 5380, 5384, 5392, 5408, 5440, 5504, 5633, 5634, 5636, 5640,
        5648, 5664, 5696, 5760, 5888, 6147, 6149, 6150, 6153, 6154, 6156, 6161, 6162, 6164, 6168,
        6177, 6178, 6180, 6184, 6192, 6209, 6210, 6212, 6216, 6224, 6240, 6273, 6274, 6276, 6280,
        6288, 6304, 6336, 6401, 6402, 6404, 6408, 6416, 6432, 6464, 6528, 6657, 6658, 6660, 6664,
        6672, 6688, 6720, 6784, 6912, 7169, 7170, 7172, 7176, 7184, 7200, 7232, 7296, 7424, 7680,
        8199, 8203, 8205, 8206, 8211, 8213, 8214, 8217, 8218, 8220, 8227, 8229, 8230, 8233, 8234,
        8236, 8241, 8242, 8244, 8248, 8259, 8261, 8262, 8265, 8266, 8268, 8273, 8274, 8276, 8280,
        8289, 8290, 8292, 8296, 8304, 8323, 8325, 8326, 8329, 8330, 8332, 8337, 8338, 8340, 8344,
        8353, 8354, 8356, 8360, 8368, 8385, 8386, 8388, 8392, 8400, 8416, 8451, 8453, 8454, 8457,
        8458, 8460, 8465, 8466, 8468, 8472, 8481, 8482, 8484, 8488, 8496, 8513, 8514, 8516, 8520,
        8528, 8544, 8577, 8578, 8580, 8584, 8592, 8608, 8640, 8707, 8709, 8710, 8713, 8714, 8716,
        8721, 8722, 8724, 8728, 8737, 8738, 8740, 8744, 8752, 8769, 8770, 8772, 8776, 8784, 8800,
        8833, 8834, 8836, 8840, 8848, 8864, 8896, 8961, 8962, 8964, 8968, 8976, 8992, 9024, 9088,
        9219, 9221, 9222, 9225, 9226, 9228, 9233, 9234, 9236, 9240, 9249, 9250, 9252, 9256, 9264,
        9281, 9282, 9284, 9288, 9296, 9312, 9345, 9346, 9348, 9352, 9360, 9376, 9408, 9473, 9474,
        9476, 9480, 9488, 9504, 9536, 9600, 9729, 9730, 9732, 9736, 9744, 9760, 9792, 9856, 9984,
        10243, 10245, 10246, 10249, 10250, 10252, 10257, 10258, 10260, 10264, 10273, 10274, 10276,
        10280, 10288, 10305, 10306, 10308, 10312, 10320, 10336, 10369, 10370, 10372, 10376, 10384,
        10400, 10432, 10497, 10498, 10500, 10504, 10512, 10528, 10560, 10624, 10753, 10754, 10756,
        10760, 10768, 10784, 10816, 10880, 11008, 11265, 11266, 11268, 11272, 11280, 11296, 11328,
        11392, 11520, 11776, 12291, 12293, 12294, 12297, 12298, 12300, 12305, 12306, 12308, 12312,
        12321, 12322, 12324, 12328, 12336, 12353, 12354, 12356, 12360, 12368, 12384, 12417, 12418,
        12420, 12424, 12432, 12448, 12480, 12545, 12546, 12548, 12552, 12560, 12576, 12608, 12672,
        12801, 12802, 12804, 12808, 12816, 12832, 12864, 12928, 13056, 13313, 13314, 13316, 13320,
        13328, 13344, 13376, 13440, 13568, 13824, 14337, 14338, 14340, 14344, 14352, 14368, 14400,
        14464, 14592, 14848, 15360, 16391, 16395, 16397, 16398, 16403, 16405, 16406, 16409, 16410,
        16412, 16419, 16421, 16422, 16425, 16426, 16428, 16433, 16434, 16436, 16440, 16451, 16453,
        16454, 16457, 16458, 16460, 16465, 16466, 16468, 16472, 16481, 16482, 16484, 16488, 16496,
        16515, 16517, 16518, 16521, 16522, 16524, 16529, 16530, 16532, 16536, 16545, 16546, 16548,
        16552, 16560, 16577, 16578, 16580, 16584, 16592, 16608, 16643, 16645, 16646, 16649, 16650,
        16652, 16657, 16658, 16660, 16664, 16673, 16674, 16676, 16680, 16688, 16705, 16706, 16708,
        16712, 16720, 16736, 16769, 16770, 16772, 16776, 16784, 16800, 16832, 16899, 16901, 16902,
        16905, 16906, 16908, 16913, 16914, 16916, 16920, 16929, 16930, 16932, 16936, 16944, 16961,
        16962, 16964, 16968, 16976, 16992, 17025, 17026, 17028, 17032, 17040, 17056, 17088, 17153,
        17154, 17156, 17160, 17168, 17184, 17216, 17280, 17411, 17413, 17414, 17417, 17418, 17420,
        17425, 17426, 17428, 17432, 17441, 17442, 17444, 17448, 17456, 17473, 17474, 17476, 17480,
        17488, 17504, 17537, 17538, 17540, 17544, 17552, 17568, 17600, 17665, 17666, 17668, 17672,
        17680, 17696, 17728, 17792, 17921, 17922, 17924, 17928, 17936, 17952, 17984, 18048, 18176,
        18435, 18437, 18438, 18441, 18442, 18444, 18449, 18450, 18452, 18456, 18465, 18466, 18468,
        18472, 18480, 18497, 18498, 18500, 18504, 18512, 18528, 18561, 18562, 18564, 18568, 18576,
        18592, 18624, 18689, 18690, 18692, 18696, 18704, 18720, 18752, 18816, 18945, 18946, 18948,
        18952, 18960, 18976, 19008, 19072, 19200, 19457, 19458, 19460, 19464, 19472, 19488, 19520,
        19584, 19712, 19968, 20483, 20485, 20486, 20489, 20490, 20492, 20497, 20498, 20500, 20504,
        20513, 20514, 20516, 20520, 20528, 20545, 20546, 20548, 20552, 20560, 20576, 20609, 20610,
        20612, 20616, 20624, 20640, 20672, 20737, 20738, 20740, 20744, 20752, 20768, 20800, 20864,
        20993, 20994, 20996, 21000, 21008, 21024, 21056, 21120, 21248, 21505, 21506, 21508, 21512,
        21520, 21536, 21568, 21632, 21760, 22016, 22529, 22530, 22532, 22536, 22544, 22560, 22592,
        22656, 22784, 23040, 23552, 24579, 24581, 24582, 24585, 24586, 24588, 24593, 24594, 24596,
        24600, 24609, 24610, 24612, 24616, 24624, 24641, 24642, 24644, 24648, 24656, 24672, 24705,
        24706, 24708, 24712, 24720, 24736, 24768, 24833, 24834, 24836, 24840, 24848, 24864, 24896,
        24960, 25089, 25090, 25092, 25096, 25104, 25120, 25152, 25216, 25344, 25601, 25602, 25604,
        25608, 25616, 25632, 25664, 25728, 25856, 26112, 26625, 26626, 26628, 26632, 26640, 26656,
        26688, 26752, 26880, 27136, 27648, 28673, 28674, 28676, 28680, 28688, 28704, 28736, 28800,
        28928, 29184, 29696, 30720, 32775, 32779, 32781, 32782, 32787, 32789, 32790, 32793, 32794,
        32796, 32803, 32805, 32806, 32809, 32810, 32812, 32817, 32818, 32820, 32824, 32835, 32837,
        32838, 32841, 32842, 32844, 32849, 32850, 32852, 32856, 32865, 32866, 32868, 32872, 32880,
        32899, 32901, 32902, 32905, 32906, 32908, 32913, 32914, 32916, 32920, 32929, 32930, 32932,
        32936, 32944, 32961, 32962, 32964, 32968, 32976, 32992, 33027, 33029, 33030, 33033, 33034,
        33036, 33041, 33042, 33044, 33048, 33057, 33058, 33060, 33064, 33072, 33089, 33090, 33092,
        33096, 33104, 33120, 33153, 33154, 33156, 33160, 33168, 33184, 33216, 33283, 33285, 33286,
        33289, 33290, 33292, 33297, 33298, 33300, 33304, 33313, 33314, 33316, 33320, 33328, 33345,
        33346, 33348, 33352, 33360, 33376, 33409, 33410, 33412, 33416, 33424, 33440, 33472, 33537,
        33538, 33540, 33544, 33552, 33568, 33600, 33664, 33795, 33797, 33798, 33801, 33802, 33804,
        33809, 33810, 33812, 33816, 33825, 33826, 33828, 33832, 33840, 33857, 33858, 33860, 33864,
        33872, 33888, 33921, 33922, 33924, 33928, 33936, 33952, 33984, 34049, 34050, 34052, 34056,
        34064, 34080, 34112, 34176, 34305, 34306, 34308, 34312, 34320, 34336, 34368, 34432, 34560,
        34819, 34821, 34822, 34825, 34826, 34828, 34833, 34834, 34836, 34840, 34849, 34850, 34852,
        34856, 34864, 34881, 34882, 34884, 34888, 34896, 34912, 34945, 34946, 34948, 34952, 34960,
        34976, 35008, 35073, 35074, 35076, 35080, 35088, 35104, 35136, 35200, 35329, 35330, 35332,
        35336, 35344, 35360, 35392, 35456, 35584, 35841, 35842, 35844, 35848, 35856, 35872, 35904,
        35968, 36096, 36352, 36867, 36869, 36870, 36873, 36874, 36876, 36881, 36882, 36884, 36888,
        36897, 36898, 36900, 36904, 36912, 36929, 36930, 36932, 36936, 36944, 36960, 36993, 36994,
        36996, 37000, 37008, 37024, 37056, 37121, 37122, 37124, 37128, 37136, 37152, 37184, 37248,
        37377, 37378, 37380, 37384, 37392, 37408, 37440, 37504, 37632, 37889, 37890, 37892, 37896,
        37904, 37920, 37952, 38016, 38144, 38400, 38913, 38914, 38916, 38920, 38928, 38944, 38976,
        39040, 39168, 39424, 39936, 40963, 40965, 40966, 40969, 40970, 40972, 40977, 40978, 40980,
        40984, 40993, 40994, 40996, 41000, 41008, 41025, 41026, 41028, 41032, 41040, 41056, 41089,
        41090, 41092, 41096, 41104, 41120, 41152, 41217, 41218, 41220, 41224, 41232, 41248, 41280,
        41344, 41473, 41474, 41476, 41480, 41488, 41504, 41536, 41600, 41728, 41985, 41986, 41988,
        41992, 42000, 42016, 42048, 42112, 42240, 42496, 43009, 43010, 43012, 43016, 43024, 43040,
        43072, 43136, 43264, 43520, 44032, 45057, 45058, 45060, 45064, 45072, 45088, 45120, 45184,
        45312, 45568, 46080, 47104, 49155, 49157, 49158, 49161, 49162, 49164, 49169, 49170, 49172,
        49176, 49185, 49186, 49188, 49192, 49200, 49217, 49218, 49220, 49224, 49232, 49248, 49281,
        49282, 49284, 49288, 49296, 49312, 49344, 49409, 49410, 49412, 49416, 49424, 49440, 49472,
        49536, 49665, 49666, 49668, 49672, 49680, 49696, 49728, 49792, 49920, 50177, 50178, 50180,
        50184, 50192, 50208, 50240, 50304, 50432, 50688, 51201, 51202, 51204, 51208, 51216, 51232,
        51264, 51328, 51456, 51712, 52224, 53249, 53250, 53252, 53256, 53264, 53280, 53312, 53376,
        53504, 53760, 54272, 55296, 57345, 57346, 57348, 57352, 57360, 57376, 57408, 57472, 57600,
        57856, 58368, 59392, 61440,
        // 228 padding entries, randomly sampled from the 1820 above
        30, 43, 106, 135, 142, 163, 267, 278, 281, 389, 449, 523, 540, 547, 564, 593, 609, 610, 612,
        649, 650, 774, 780, 898, 1050, 1052, 1094, 1097, 1108, 1121, 1200, 1220, 1232, 1314, 1320,
        1412, 1424, 1542, 1569, 1576, 1824, 2098, 2104, 2145, 2148, 2185, 2188, 2370, 2625, 2626,
        2690, 2752, 3089, 3152, 3204, 3344, 4107, 4115, 4131, 4152, 4194, 4227, 4236, 4241, 4257,
        4260, 4355, 4400, 4484, 4544, 4614, 4617, 4648, 4737, 4872, 4880, 5125, 5186, 5216, 5256,
        6149, 6154, 6161, 6164, 6180, 6276, 6528, 6658, 8203, 8205, 8213, 8218, 8266, 8280, 8400,
        8416, 8451, 8453, 8457, 8721, 8738, 8834, 8840, 9226, 9264, 9282, 9346, 9504, 9536, 9736,
        10264, 10305, 10400, 12297, 12548, 12576, 12804, 13314, 13328, 13376, 14368, 16397, 16409,
        16466, 16472, 16481, 16488, 16517, 16545, 16546, 16584, 16646, 16650, 16658, 16936, 17056,
        17474, 17488, 17668, 17922, 17924, 18576, 18704, 18976, 19460, 20486, 20513, 20752, 20768,
        24581, 24594, 24596, 24624, 24896, 25092, 25120, 25601, 26628, 27136, 32775, 32782, 32787,
        32805, 32812, 32818, 32842, 32850, 32872, 32880, 32901, 32902, 32913, 32914, 32962, 32976,
        32992, 33072, 33289, 33328, 33410, 33412, 33424, 33540, 33600, 33795, 33809, 33816, 33840,
        33888, 33936, 34056, 34112, 34819, 34828, 34836, 36881, 36884, 36904, 36930, 37152, 37380,
        38914, 38944, 39424, 40963, 40977, 41025, 41090, 41120, 41152, 41480, 42240, 43016, 46080,
        49192, 49200, 49440, 49472, 49668, 50178, 51204, 51216, 51456, 53250, 57345, 57352, 57376,
        57856,
    ],
};

#[inline(always)]
fn bloom_get_tag(hash: u32) -> u16 {
    BLOOM_TAGS.data[(hash >> (32 - 11)) as usize]
}

#[inline(always)]
fn bloom_check_tag(tag: u16, entry: u16) -> bool {
    (entry & tag) == tag
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct HashPair {
    pub slot: u64,
    pub filter: u32,
}

impl HashPair {
    const FIBONACCI: u64 = 11_400_714_819_323_198_485;

    #[inline(always)]
    pub fn hash(key: u32) -> Self {
        let v = (key as u64).wrapping_mul(Self::FIBONACCI);
        Self {
            slot: v,
            filter: v as u32,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct DirectoryEntry(u64);

impl DirectoryEntry {
    pub const EMPTY: Self = Self(0);
    const OFFSET_SHIFT: u32 = 16;

    #[inline(always)]
    pub fn new(offset: u64, bloom: u16) -> Self {
        Self((offset << Self::OFFSET_SHIFT) | bloom as u64)
    }

    #[inline(always)]
    pub fn offset(self) -> u64 {
        self.0 >> Self::OFFSET_SHIFT
    }

    #[inline(always)]
    pub fn bloom(self) -> u16 {
        self.0 as u16
    }

    #[inline(always)]
    pub fn with_tag(self, tag: u16) -> Self {
        Self(self.0 | tag as u64)
    }

    #[inline(always)]
    pub fn add_offset(self, byte_delta: u64) -> Self {
        Self(self.0.wrapping_add(byte_delta << Self::OFFSET_SHIFT))
    }
}

/// Compute directory size (power-of-two slots) and shift for slot selection.
/// Table is sized to ~1.125n giving ~65% load factor. Minimum 16 slots.
fn compute_table_params(num_tuples: usize) -> (usize, u32) {
    let min_size = 16_usize;
    let target = (num_tuples + (num_tuples / 8)).max(min_size);
    let table_size = target.next_power_of_two();
    let shift = 64 - table_size.trailing_zeros();
    (table_size, shift)
}

pub struct UnchainedHashTable {
    directory: Vec<DirectoryEntry>,
    tuple_storage: Vec<u8>,
    shift: u32,
    tuple_stride: usize,
}

impl UnchainedHashTable {
    pub fn empty(tuple_stride: usize) -> Self {
        let (table_size, shift) = compute_table_params(0);
        Self {
            directory: vec![DirectoryEntry::EMPTY; table_size + 1],
            tuple_storage: Vec::new(),
            shift,
            tuple_stride,
        }
    }

    /// Probe for a key, calling `callback` for each matching tuple.
    /// Returns true if the Bloom filter indicated a possible match.
    #[inline(always)]
    pub fn probe(&self, key: u32, mut callback: impl FnMut(&[u64])) -> bool {
        let h = HashPair::hash(key);
        let slot = (h.slot >> self.shift) as usize;
        let entry = self.directory[slot + 1];

        let tag = bloom_get_tag(h.filter);
        if !bloom_check_tag(tag, entry.bloom()) {
            return false;
        }

        let start = self.directory[slot].offset() as usize;
        let end = entry.offset() as usize;
        let fields_per_tuple = self.tuple_stride / size_of::<u64>();

        let mut pos = start;
        while pos < end {
            unsafe {
                let base = self.tuple_storage.as_ptr();
                let tuple_ptr = base.add(pos) as *const u64;
                let tuple_key = *tuple_ptr;

                if tuple_key == key as u64 {
                    let tuple_slice = std::slice::from_raw_parts(tuple_ptr, fields_per_tuple);
                    callback(tuple_slice);
                }
            }
            pos += self.tuple_stride;
        }

        true
    }

    /// Bloom filter check only — useful as a semi-join reducer pushed
    /// into earlier operators in the query pipeline.
    #[inline(always)]
    pub fn bloom_check(&self, key: u32) -> bool {
        let h = HashPair::hash(key);
        let slot = (h.slot >> self.shift) as usize;
        let entry = self.directory[slot + 1];
        let tag = bloom_get_tag(h.filter);
        bloom_check_tag(tag, entry.bloom())
    }

    pub fn num_tuples(&self) -> usize {
        if self.tuple_stride == 0 {
            0
        } else {
            self.tuple_storage.len() / self.tuple_stride
        }
    }
}

/// Wrapper to send raw pointers across thread boundaries.
///
/// Safety: callers must ensure threads write to disjoint memory regions.
#[derive(Copy, Clone)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

pub struct BuildConfig {
    pub num_partitions_shift: u32,
    pub tuple_stride: usize,
}

impl BuildConfig {
    pub fn new(tuple_stride: usize) -> Self {
        debug_assert!(tuple_stride >= size_of::<u64>());
        debug_assert!(tuple_stride % size_of::<u64>() == 0);
        Self {
            num_partitions_shift: 7,
            tuple_stride,
        }
    }

    pub fn with_partitions(tuple_stride: usize, partitions_shift: u32) -> Self {
        let mut c = Self::new(tuple_stride);
        c.num_partitions_shift = partitions_shift;
        c
    }

    fn num_partitions(&self) -> usize {
        1 << self.num_partitions_shift
    }

    fn partition_shift(&self) -> u32 {
        64 - self.num_partitions_shift
    }
}

pub struct LocalCollector {
    buffers: Vec<Vec<u8>>,
    tuple_count: usize,
    partition_shift: u32,
    tuple_stride: usize,
}

impl LocalCollector {
    pub fn new(config: &BuildConfig) -> Self {
        Self {
            buffers: (0..config.num_partitions()).map(|_| Vec::new()).collect(),
            tuple_count: 0,
            partition_shift: config.partition_shift(),
            tuple_stride: config.tuple_stride,
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, key: u32, payload: &[u64]) {
        debug_assert_eq!(size_of::<u64>() * (1 + payload.len()), self.tuple_stride);

        let h = HashPair::hash(key);
        let partition = (h.slot >> self.partition_shift) as usize;

        let buf = &mut self.buffers[partition];
        buf.extend_from_slice(&(key as u64).to_ne_bytes());
        for &val in payload {
            buf.extend_from_slice(&val.to_ne_bytes());
        }
        self.tuple_count += 1;
    }

    pub fn tuple_count(&self) -> usize {
        self.tuple_count
    }
}

/// Build an UnchainedHashTable from collected tuples.
///
/// Three-phase parallel build:
/// 1. Count per slot + Bloom tags (parallel by partition)
/// 2. Exclusive prefix sum (sequential, O(table_size))
/// 3. Copy tuples to final storage (parallel by partition)
pub fn build(collectors: Vec<LocalCollector>, config: &BuildConfig) -> UnchainedHashTable {
    let total_tuples: usize = collectors.iter().map(|c| c.tuple_count).sum();
    let stride = config.tuple_stride;

    if total_tuples == 0 {
        return UnchainedHashTable::empty(stride);
    }

    let (table_size, shift) = compute_table_params(total_tuples);

    let collector_partitions = config.num_partitions();
    let num_partitions = collector_partitions.min(table_size);
    let merge_factor = collector_partitions / num_partitions;

    // Merge per-partition buffers from all collectors
    let partition_data: Vec<Vec<u8>> = (0..num_partitions)
        .map(|ep| {
            let mut merged = Vec::new();
            for orig_p in (ep * merge_factor)..((ep + 1) * merge_factor) {
                for c in &collectors {
                    merged.extend_from_slice(&c.buffers[orig_p]);
                }
            }
            merged
        })
        .collect();

    let mut directory = vec![DirectoryEntry::EMPTY; table_size + 1];
    let total_bytes = total_tuples * stride;
    let mut tuple_storage = vec![0u8; total_bytes];

    // Phase 1: Count per slot + accumulate Bloom tags
    {
        let dir_ptr = SendPtr(directory.as_mut_ptr());
        thread::scope(|s| {
            for p in 0..num_partitions {
                let data = &partition_data[p];
                let dir_ptr = SendPtr(dir_ptr.0);
                s.spawn(move || {
                    let mut pos = 0;
                    while pos + stride <= data.len() {
                        let key_bytes: [u8; 8] = data[pos..pos + 8].try_into().unwrap();
                        let key = u64::from_ne_bytes(key_bytes) as u32;
                        let h = HashPair::hash(key);
                        let slot = (h.slot >> shift) as usize;
                        let tag = bloom_get_tag(h.filter);
                        unsafe {
                            let entry = &mut *dir_ptr.get().add(slot + 1);
                            *entry = entry.add_offset(stride as u64).with_tag(tag);
                        }
                        pos += stride;
                    }
                });
            }
        });
    }

    // Phase 2: Exclusive prefix sum
    {
        let mut cumulative: u64 = 0;
        for i in 1..directory.len() {
            let count = directory[i].offset();
            directory[i] = DirectoryEntry::new(cumulative, directory[i].bloom());
            cumulative += count;
        }
        debug_assert_eq!(cumulative, total_bytes as u64);
    }

    // Phase 3: Copy tuples to final storage
    {
        let dir_ptr = SendPtr(directory.as_mut_ptr());
        let store_ptr = SendPtr(tuple_storage.as_mut_ptr());
        thread::scope(|s| {
            for p in 0..num_partitions {
                let data = &partition_data[p];
                let dir_ptr = SendPtr(dir_ptr.0);
                let store_ptr = SendPtr(store_ptr.0);
                s.spawn(move || {
                    let mut pos = 0;
                    while pos + stride <= data.len() {
                        let key_bytes: [u8; 8] = data[pos..pos + 8].try_into().unwrap();
                        let key = u64::from_ne_bytes(key_bytes) as u32;
                        let h = HashPair::hash(key);
                        let slot = (h.slot >> shift) as usize;
                        unsafe {
                            let entry = &mut *dir_ptr.get().add(slot + 1);
                            let cursor = entry.offset() as usize;
                            std::ptr::copy_nonoverlapping(
                                data[pos..].as_ptr(),
                                store_ptr.get().add(cursor),
                                stride,
                            );
                            *entry = entry.add_offset(stride as u64);
                        }
                        pos += stride;
                    }
                });
            }
        });
    }

    UnchainedHashTable {
        directory,
        tuple_storage,
        shift,
        tuple_stride: stride,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRIDE: usize = 16; // key (u64) + one payload (u64)

    // -- Bloom filter tests -------------------------------------------------

    #[test]
    fn bloom_table_all_popcount_4() {
        for (i, &mask) in BLOOM_TAGS.data.iter().enumerate() {
            assert_eq!(mask.count_ones(), 4, "entry {i} = {mask:#06x}");
        }
    }

    #[test]
    fn bloom_table_alignment() {
        let addr = &BLOOM_TAGS as *const BloomTable as usize;
        assert_eq!(addr % 4096, 0);
    }

    #[test]
    fn bloom_first_1820_distinct() {
        let mut seen = std::collections::HashSet::new();
        for &mask in &BLOOM_TAGS.data[..1820] {
            assert!(seen.insert(mask));
        }
    }

    #[test]
    fn bloom_self_check() {
        for &mask in BLOOM_TAGS.data.iter() {
            assert!(bloom_check_tag(mask, mask));
        }
    }

    #[test]
    fn bloom_empty_rejects() {
        for &mask in BLOOM_TAGS.data.iter() {
            assert!(!bloom_check_tag(mask, 0));
        }
    }

    // -- Hash tests ---------------------------------------------------------

    #[test]
    fn hash_zero() {
        let h = HashPair::hash(0);
        assert_eq!(h.slot, 0);
        assert_eq!(h.filter, 0);
    }

    #[test]
    fn hash_filter_is_low_bits() {
        for key in 0..10_000 {
            let h = HashPair::hash(key);
            assert_eq!(h.filter, h.slot as u32);
        }
    }

    #[test]
    fn hash_no_catastrophic_collisions() {
        let mut seen = std::collections::HashSet::new();
        for key in 0..10_000_u32 {
            seen.insert(HashPair::hash(key).slot);
        }
        assert!(seen.len() > 9_900);
    }

    // -- Directory entry tests ----------------------------------------------

    #[test]
    fn directory_round_trip() {
        let e = DirectoryEntry::new(123456, 0xABCD);
        assert_eq!(e.offset(), 123456);
        assert_eq!(e.bloom(), 0xABCD);
    }

    #[test]
    fn directory_with_tag_preserves_offset() {
        let e = DirectoryEntry::new(999, 0).with_tag(0b1111);
        assert_eq!(e.offset(), 999);
        assert_eq!(e.bloom(), 0b1111);
    }

    #[test]
    fn directory_add_offset_preserves_bloom() {
        let e = DirectoryEntry::new(100, 0xFFFF).add_offset(50);
        assert_eq!(e.offset(), 150);
        assert_eq!(e.bloom(), 0xFFFF);
    }

    #[test]
    fn sizing_basics() {
        let (size, shift) = compute_table_params(0);
        assert_eq!(size, 16);
        assert_eq!(shift, 60);

        for n in [100, 10_000, 1_000_000] {
            let (size, shift) = compute_table_params(n);
            assert!(size.is_power_of_two());
            assert!(size >= n);
            assert_eq!(1_usize << (64 - shift), size);
        }
    }

    // -- Probe tests (manual build) -----------------------------------------

    fn build_test_table(tuples: &[(u32, u64)]) -> UnchainedHashTable {
        if tuples.is_empty() {
            return UnchainedHashTable::empty(STRIDE);
        }
        let (table_size, shift) = compute_table_params(tuples.len());
        let mut directory = vec![DirectoryEntry::EMPTY; table_size + 1];

        for &(key, _) in tuples {
            let h = HashPair::hash(key);
            let slot = (h.slot >> shift) as usize;
            let tag = bloom_get_tag(h.filter);
            directory[slot + 1] = directory[slot + 1].add_offset(STRIDE as u64).with_tag(tag);
        }

        let mut cumulative: u64 = 0;
        for i in 1..directory.len() {
            let count = directory[i].offset();
            directory[i] = DirectoryEntry::new(cumulative, directory[i].bloom());
            cumulative += count;
        }

        let mut tuple_storage = vec![0u8; cumulative as usize];
        for &(key, payload) in tuples {
            let h = HashPair::hash(key);
            let slot = (h.slot >> shift) as usize;
            let cursor = directory[slot + 1].offset() as usize;
            tuple_storage[cursor..cursor + 8].copy_from_slice(&(key as u64).to_ne_bytes());
            tuple_storage[cursor + 8..cursor + 16].copy_from_slice(&payload.to_ne_bytes());
            directory[slot + 1] = directory[slot + 1].add_offset(STRIDE as u64);
        }

        UnchainedHashTable {
            directory,
            tuple_storage,
            shift,
            tuple_stride: STRIDE,
        }
    }

    #[test]
    fn probe_single_match() {
        let table = build_test_table(&[(42, 100)]);
        let mut found = Vec::new();
        table.probe(42, |t| found.push((t[0], t[1])));
        assert_eq!(found, vec![(42, 100)]);
    }

    #[test]
    fn probe_absent_key() {
        let table = build_test_table(&[(42, 100)]);
        let mut found = false;
        table.probe(99, |_| found = true);
        assert!(!found);
    }

    #[test]
    fn probe_duplicates() {
        let table = build_test_table(&[(10, 1), (10, 2), (20, 3)]);
        let mut found = Vec::new();
        table.probe(10, |t| found.push(t[1]));
        found.sort();
        assert_eq!(found, vec![1, 2]);
    }

    #[test]
    fn probe_all_keys() {
        let data: Vec<(u32, u64)> = (0..100).map(|i| (i, i as u64 * 10)).collect();
        let table = build_test_table(&data);
        for &(key, payload) in &data {
            let mut found = false;
            table.probe(key, |t| {
                assert_eq!(t[0], key as u64);
                assert_eq!(t[1], payload);
                found = true;
            });
            assert!(found, "key {key} not found");
        }
    }

    // -- Parallel build tests -----------------------------------------------

    fn verify_all_present(table: &UnchainedHashTable, tuples: &[(u32, u64)]) {
        let mut expected: std::collections::HashMap<u32, Vec<u64>> =
            std::collections::HashMap::new();
        for &(key, payload) in tuples {
            expected.entry(key).or_default().push(payload);
        }
        for (&key, expected_payloads) in &expected {
            let mut found = Vec::new();
            table.probe(key, |t| {
                assert_eq!(t[0], key as u64);
                found.push(t[1]);
            });
            found.sort();
            let mut sorted = expected_payloads.clone();
            sorted.sort();
            assert_eq!(found, sorted, "mismatch for key {key}");
        }
    }

    fn build_single(tuples: &[(u32, u64)]) -> UnchainedHashTable {
        let config = BuildConfig::new(STRIDE);
        let mut c = LocalCollector::new(&config);
        for &(key, payload) in tuples {
            c.insert(key, &[payload]);
        }
        build(vec![c], &config)
    }

    fn build_multi(tuples: &[(u32, u64)], num_threads: usize) -> UnchainedHashTable {
        let config = BuildConfig::new(STRIDE);
        let collectors: Vec<LocalCollector> = thread::scope(|s| {
            let chunk_size = (tuples.len() + num_threads - 1) / num_threads;
            let handles: Vec<_> = tuples
                .chunks(chunk_size)
                .map(|chunk| {
                    let config = &config;
                    s.spawn(move || {
                        let mut c = LocalCollector::new(config);
                        for &(key, payload) in chunk {
                            c.insert(key, &[payload]);
                        }
                        c
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        build(collectors, &config)
    }

    #[test]
    fn build_empty() {
        let config = BuildConfig::new(STRIDE);
        let c = LocalCollector::new(&config);
        let table = build(vec![c], &config);
        let mut found = false;
        table.probe(42, |_| found = true);
        assert!(!found);
    }

    #[test]
    fn build_single_tuple() {
        let table = build_single(&[(42, 100)]);
        verify_all_present(&table, &[(42, 100)]);
    }

    #[test]
    fn build_no_false_negatives() {
        let data: Vec<(u32, u64)> = (0..1000).map(|i| (i, i as u64 * 10)).collect();
        let table = build_single(&data);
        verify_all_present(&table, &data);
    }

    #[test]
    fn build_duplicates() {
        let data = vec![(10, 1), (10, 2), (10, 3), (20, 4), (20, 5), (30, 6)];
        let table = build_single(&data);
        verify_all_present(&table, &data);
    }

    #[test]
    fn build_absent_keys() {
        let data: Vec<(u32, u64)> = (0..100).map(|i| (i * 2, 0)).collect();
        let table = build_single(&data);
        for i in 0..100 {
            let mut found = false;
            table.probe(i * 2 + 1, |_| found = true);
            assert!(!found, "false match for key {}", i * 2 + 1);
        }
    }

    #[test]
    fn build_multi_thread() {
        let data: Vec<(u32, u64)> = (0..5000).map(|i| (i, i as u64 * 7)).collect();
        let table = build_multi(&data, 4);
        verify_all_present(&table, &data);
    }

    #[test]
    fn build_multi_thread_duplicates() {
        let data: Vec<(u32, u64)> = (0..5000).map(|i| ((i % 1000) as u32, i as u64)).collect();
        let table = build_multi(&data, 8);
        verify_all_present(&table, &data);
    }

    #[test]
    fn build_multi_thread_many_threads() {
        let data: Vec<(u32, u64)> = (0..10000).map(|i| (i, i as u64)).collect();
        let table = build_multi(&data, 16);
        verify_all_present(&table, &data);
    }

    #[test]
    fn build_bloom_no_false_negatives() {
        let data: Vec<(u32, u64)> = (0..10000).map(|i| (i, 0)).collect();
        let table = build_single(&data);
        for &(key, _) in &data {
            assert!(table.bloom_check(key), "bloom rejected key {key}");
        }
    }

    #[test]
    fn build_tuple_count() {
        let config = BuildConfig::new(STRIDE);
        let mut c = LocalCollector::new(&config);
        for i in 0..777_u32 {
            c.insert(i, &[i as u64]);
        }
        assert_eq!(c.tuple_count(), 777);
        let table = build(vec![c], &config);
        assert_eq!(table.num_tuples(), 777);
    }

    #[test]
    fn build_small_partition_count() {
        let config = BuildConfig::with_partitions(STRIDE, 2);
        let mut c = LocalCollector::new(&config);
        for i in 0..500_u32 {
            c.insert(i, &[i as u64 * 3]);
        }
        let table = build(vec![c], &config);
        for i in 0..500_u32 {
            let mut found = false;
            table.probe(i, |t| {
                assert_eq!(t[1], i as u64 * 3);
                found = true;
            });
            assert!(found, "key {i} not found");
        }
    }
}
