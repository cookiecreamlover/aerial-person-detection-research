"""Guardrails for scientifically valid modality claims."""

def validate_manifest(rows, modality, thermal_required=True):
    if modality not in {'rgb','thermal','fused'}: raise ValueError(f'Unknown modality: {modality}')
    if modality == 'rgb': return
    if thermal_required:
        missing=[r.get('image_id',r.get('image')) for r in rows if not r.get('thermal_image')]
        if missing:
            raise ValueError('Thermal/fused evaluation requires synchronized thermal_image paths; RGB-only COCO rows cannot be used as thermal evidence.')
        if modality == 'fused' and any(not r.get('rgb_image') for r in rows):
            raise ValueError('Fused evaluation requires both rgb_image and thermal_image in every manifest row.')

def modality_note(modality):
    return {'rgb':'real_rgb_evaluation','thermal':'real_thermal_evaluation','fused':'real_fused_evaluation'}[modality]
