# NOTE: model_id = index in MODELS
MODELS = [
    {
        "MODEL_NAME": "text_encoder",
        "MODEL_SIZE": 5100000, # 5677000,       # in kB
        "MAX_BATCH_SIZE": 4,
        "BATCH_SIZES": [1, 2, 3, 4],

        # concurrency
        # "BATCH_EXEC_TIME": [9.15, 12.16, 16.57, 21.50],
        # "MIG_BATCH_EXEC_TIMES": {6: [9.15, 12.16, 16.57, 21.50],
        #                          24: [9.15, 12.16, 16.57, 21.50]},

        # no concurrency
        "BATCH_EXEC_TIME": [7.10, 8.38, 10.29, 13.16],
        "MIG_BATCH_EXEC_TIMES": {6: [7.10, 8.38, 10.29, 13.16],
                                    24: [7.10, 8.38, 10.29, 13.16]},
        "EXEC_TIME_CV": 0.06
    },
    {
        "MODEL_NAME": "vision_encoder",
        "MODEL_SIZE": 20919000,      # in kB
        "MAX_BATCH_SIZE": 16,
        "BATCH_SIZES": list(range(1,17)),
        "MIG_BATCH_EXEC_TIMES": {24: [30.57, 54.38, 77.24, 101.91, 
                                        127.34, 150.61, 166.60, 189.00, 215.82, 
                                        231.59, 253.35, 265.56, 292.49, 313.73,
                                        341.07, 358.86]},
        "EXEC_TIME_CV": 0.05
    },
    {
        "MODEL_NAME": "flmr",
        "MODEL_SIZE": 854000,        # in KB
        "MAX_BATCH_SIZE": 8,
        "BATCH_SIZES": list(range(1,9)),
        "MIG_BATCH_EXEC_TIMES": {
            6: [5.35, 9.45, 13.64, 17.62, 21.70, 26.78, 31.08, 34.30],
            24: [5.35, 9.45, 13.64, 17.62, 21.70, 26.78, 31.08, 34.30]
        }, # { 6: [8.94, 15.15, 21.92, 27.91, 31.86, 38.95, 45.69, 53.84]},
        "EXEC_TIME_CV": 0.95
    },
    {
        "MODEL_NAME": "search",
        "MODEL_SIZE": 777000,        # in KB
        "MAX_BATCH_SIZE": 16,
        "BATCH_SIZES": list(range(1,17)),
        "MIG_BATCH_EXEC_TIMES": {6: [24.56, 48.38, 66.22, 86.08,
                                        108.49, 136.14, 153.81, 175.46,
                                        200.74, 228.27, 245.03, 259.32,
                                        283.92, 307.50, 329.19, 347.88],
                                24: [24.56, 48.38, 66.22, 86.08,
                                        108.49, 136.14, 153.81, 175.46,
                                        200.74, 228.27, 245.03, 259.32,
                                        283.92, 307.50, 329.19, 347.88]},
        "EXEC_TIME_CV": 0.06
    },
    {
        "MODEL_NAME": "audio_det",
        "MODEL_SIZE": 6093000,      # in kB
        "MAX_BATCH_SIZE": 8,
        "BATCH_SIZES": [1, 2, 4, 8],
        "MIG_BATCH_EXEC_TIMES": {
            24: [65.0, 68.0, 69.4, 72.1],
            12: [65.0, 68.0, 69.4, 72.1] # TODO: update with real nums
        },
        "EXEC_TIME_CV": 0.182
    },
    {
        "MODEL_NAME": "encode_search-ivf",
        "MODEL_SIZE": 1210000,       # in kB
        "MAX_BATCH_SIZE": 8,
        "BATCH_SIZES": [1, 2, 4, 8],
        "MIG_BATCH_EXEC_TIMES": {
            24: [16.7, 17.2, 17.5, 17.5], # TODO [0.397, 0.405, 0.424, 0.456],
            12: [16.7, 17.2, 17.5, 17.5],
            6: [16.5, 16.9, 16.9, 17.3]
        },
        "EXEC_TIME_CV": 0.414,
    },
    {
        "MODEL_NAME": "text_check",
        "MODEL_SIZE": 2101000,       # in kB
        "MAX_BATCH_SIZE": 2,
        "BATCH_SIZES": [1, 2],
        "MIG_BATCH_EXEC_TIMES": {
            24: [17.3, 25.3], 12: [25.3, 44.2]
        },
        "EXEC_TIME_CV": 0.65,
    },
    {
        "MODEL_NAME": "aggregate-tts",
        "MODEL_SIZE": 6135000,             # in kB
        "MAX_BATCH_SIZE": 1,
        "BATCH_SIZES": [1],
        "MIG_BATCH_EXEC_TIMES": {24: [87.3], 12: [149.3]},
        "EXEC_TIME_CV": 0.389
    },
    {
        "MODEL_NAME": "lang-detection",
        "MODEL_SIZE": 162100,           # in kB
        "MAX_BATCH_SIZE": 8,
        "BATCH_SIZES": [1, 8],
        "MIG_BATCH_EXEC_TIMES": {24: [6.3, 24.1], 12: [6.3, 24.1]},
        "EXEC_TIME_CV": 0.1
    },
    {
        "MODEL_NAME": "lang-translation",
        "MODEL_SIZE": 2000000,             # in kB
        "MAX_BATCH_SIZE": 1,
        "BATCH_SIZES": [1],
        "MIG_BATCH_EXEC_TIMES": {24: [549.9], 12: [549.9]},
        "EXEC_TIME_CV": 0.5
    }
]