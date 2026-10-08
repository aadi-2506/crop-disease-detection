from __future__ import annotations


DISEASE_GUIDES = {
    "apple scab": {
        "description": "A fungal disease that causes olive-green to dark, scabby lesions.",
        "symptoms": "Olive or dark spots on leaves and fruit; leaves may yellow and fall early.",
        "control": "Remove fallen leaves, improve airflow by pruning, and ask a local expert about suitable fungicide timing.",
    },
    "apple black rot": {
        "description": "A fungal disease that can affect apple leaves, fruit and branches.",
        "symptoms": "Small purple leaf spots that develop pale centers, or dark fruit rot.",
        "control": "Remove mummified fruit and diseased wood, keep the canopy open, and avoid overhead watering.",
    },
    "cedar apple rust": {
        "description": "A fungal disease that alternates between apple trees and nearby juniper hosts.",
        "symptoms": "Orange-yellow spots on leaves, sometimes with tube-like growths underneath.",
        "control": "Remove nearby infected galls when practical and choose resistant varieties suited to the area.",
    },
    "early blight": {
        "description": "A fungal leaf disease that commonly affects tomato and potato plants.",
        "symptoms": "Brown spots, often with concentric rings, and yellowing of older leaves.",
        "control": "Rotate crops, remove infected leaves, mulch to limit soil splash, and water at the base.",
    },
    "late blight": {
        "description": "A fast-spreading disease that can severely damage tomato and potato crops.",
        "symptoms": "Water-soaked patches that turn brown; white growth may appear around lesions in humid weather.",
        "control": "Remove infected plants promptly, avoid wet foliage, improve airflow, and seek local disease guidance.",
    },
    "bacterial spot": {
        "description": "A bacterial disease affecting leaves and fruit of several crops.",
        "symptoms": "Small dark, sometimes water-soaked spots that may develop yellow margins.",
        "control": "Use clean seed and tools, avoid handling wet plants, rotate crops, and remove infected debris.",
    },
    "leaf mold": {
        "description": "A fungal disease favored by humid conditions, especially in greenhouse tomatoes.",
        "symptoms": "Pale yellow patches above leaves and olive or gray fuzzy growth underneath.",
        "control": "Lower humidity, increase ventilation and spacing, and remove badly affected leaves.",
    },
    "septoria leaf spot": {
        "description": "A fungal disease that often starts on older tomato leaves.",
        "symptoms": "Many small round spots with pale centers and dark edges; lower leaves may die back.",
        "control": "Remove affected lower leaves, rotate crops, mulch, and avoid overhead irrigation.",
    },
    "spider mites": {
        "description": "Tiny sap-feeding pests that thrive in hot, dry conditions.",
        "symptoms": "Fine pale stippling, bronzed leaves and sometimes delicate webbing underneath.",
        "control": "Check leaf undersides, reduce plant stress, and use locally recommended integrated pest management.",
    },
    "target spot": {
        "description": "A fungal leaf disease that causes spreading spots on susceptible crops.",
        "symptoms": "Brown lesions with ring-like patterns that can merge and cause leaf drop.",
        "control": "Improve airflow, remove infected plant material, rotate crops, and keep leaves dry where possible.",
    },
    "yellow leaf curl virus": {
        "description": "A virus commonly spread by whiteflies in tomato crops.",
        "symptoms": "Upward-curling yellow leaves, stunted growth and reduced fruit set.",
        "control": "Remove infected plants, monitor whiteflies, control weeds and use resistant varieties if available.",
    },
    "mosaic virus": {
        "description": "A group of viruses that can cause mottled patterns and distorted growth.",
        "symptoms": "Light and dark green mosaic patterns, leaf distortion and reduced plant vigor.",
        "control": "Remove infected plants, disinfect tools, control insect vectors and use certified clean planting material.",
    },
    "powdery mildew": {
        "description": "A fungal disease that forms a powder-like coating on plant surfaces.",
        "symptoms": "White powdery patches on leaves or stems, sometimes followed by yellowing.",
        "control": "Space plants for airflow, remove affected foliage, and avoid excess shade and humidity.",
    },
    "rust": {
        "description": "A fungal disease that produces rust-colored spore patches on leaves.",
        "symptoms": "Orange, brown or reddish pustules, often visible on the underside of leaves.",
        "control": "Remove infected debris, improve airflow and avoid wetting foliage late in the day.",
    },
    "healthy": {
        "description": "The model found no familiar disease pattern in this image.",
        "symptoms": "No disease symptoms were identified by the model in the submitted image.",
        "control": "Continue routine monitoring and good crop care. Recheck if new spots, discoloration or wilting appear.",
    },
}

GENERIC_GUIDE = {
    "description": "The model recognized a crop disease class from its training labels.",
    "symptoms": "Look for leaf spots, discoloration, curling, wilting or unusual growth, and compare with a local expert's diagnosis.",
    "control": "Remove badly affected material where appropriate, keep tools clean, avoid excess leaf moisture and seek local agricultural advice.",
}


def get_prediction_details(class_name: str) -> tuple[str, str, str]:
    if "___" in class_name:
        crop, condition = class_name.split("___", maxsplit=1)
    elif "__" in class_name:
        crop, condition = class_name.split("__", maxsplit=1)
    elif "_" in class_name:
        crop, condition = class_name.split("_", maxsplit=1)
    else:
        crop, condition = class_name, ""
    crop = " ".join(crop.replace("_", " ").split()) or "Unknown crop"
    condition = " ".join(condition.replace("_", " ").split()) or class_name.replace("_", " ")
    status = "healthy" if "healthy" in condition.lower() else "diseased"
    return crop, condition, status


def get_disease_info(class_name: str) -> dict[str, str]:
    crop, disease, status = get_prediction_details(class_name)
    normalized = disease.lower().strip()
    guide = next(
        (
            details
            for key, details in DISEASE_GUIDES.items()
            if key in normalized
        ),
        None,
    )
    if guide is None:
        guide = DISEASE_GUIDES["healthy"] if status == "healthy" else GENERIC_GUIDE
    return {
        "name": f"{crop} {disease}".strip(),
        **guide,
    }
