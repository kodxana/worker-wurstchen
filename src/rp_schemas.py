INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': False,
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'prior_guidance_scale': {
        'type': float,
        'required': False,
        'default': 4.0
    },
    'num_images_per_prompt': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 3 > img_count > 0
    },
    'output_type': {
        'type': str,
        'required': False,
        'default': 'pil'
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 0.0
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 50
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': 'DDIM'
    }
}