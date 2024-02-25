import json
import httpx
import platform
import pygame

def sound_alarm():
    try:
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load("cheering.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        pygame.mixer.quit()
        pygame.quit()
    except Exception as e:
        print(f"Failed to play alarm sound: {e}")


API_KEY = 'AI0SVJ2XA80KY4SQ8N2JLIKXV6H1SJ9KOPT55CHY'
GPU_NAMES = ['NVIDIA RTX A2000', 'NVIDIA GeForce RTX 3070', 'NVIDIA GeForce RTX 3080', 'NVIDIA GeForce RTX 3080 Ti', 'NVIDIA GeForce RTX 4070 Ti', 'NVIDIA GeForce RTX 4080', 'NVIDIA RTX A4000', 'Tesla V100-FHHL-16GB', 'Tesla V100-PCIE-16GB', 'Tesla V100-SXM2-16GB', 'NVIDIA RTX 4000 Ada Generation', 'NVIDIA RTX 4000 SFF Ada Generation', 'NVIDIA RTX A4500', 'NVIDIA A30', 'NVIDIA GeForce RTX 3090', 'NVIDIA GeForce RTX 3090 Ti', 'NVIDIA GeForce RTX 4090', 'NVIDIA L4', 'NVIDIA RTX A5000', 'NVIDIA RTX 5000 Ada Generation', 'Tesla V100-SXM2-32GB', 'NVIDIA A40', 'NVIDIA L40', 'NVIDIA RTX 6000 Ada Generation', 'NVIDIA RTX A6000', 'NVIDIA A100 80GB PCIe', 'NVIDIA A100-SXM4-80GB', 'NVIDIA H100 80GB HBM3', 'NVIDIA H100 PCIe']


def _run_query(payload, auth_required=True):
    url = 'https://api.runpod.io/graphql'

    if auth_required:
        url += f'?api_key={API_KEY}'

    response = httpx.post(
        url,
        json=payload
    )

    return response

def get_availability(GPU_NAME, secure):
    query = {
    "operationName": "SecureGpuTypes",
    "variables": {
        "gpuTypesInput": {
        "id": GPU_NAME
        },
        "lowestPriceInput": {
            "dataCenterId": None,
            "gpuCount": 1,
            "minDisk": 0,
            "minMemoryInGb": 8,
            "minVcpuCount": 1,
            "secureCloud": secure,
            "allowedCudaVersions": [
                "11.8"
            ]
        }
    },
    "query": "query SecureGpuTypes($lowestPriceInput: GpuLowestPriceInput, $gpuTypesInput: GpuTypeFilter) {\n  gpuTypes(input: $gpuTypesInput) {\n    lowestPrice(input: $lowestPriceInput) {\n      minimumBidPrice\n      uninterruptablePrice\n      minVcpu\n      minMemory\n      stockStatus\n      __typename\n    }\n    id\n    displayName\n    memoryInGb\n    securePrice\n    communityPrice\n    oneMonthPrice\n    threeMonthPrice\n    sixMonthPrice\n    secureSpotPrice\n    __typename\n  }\n}"
    }

    response = _run_query(query).json()["data"]["gpuTypes"][0]["lowestPrice"]["stockStatus"]

    return response is not None

is_gpu_available = False

while not is_gpu_available:
    for GPU_NAME in GPU_NAMES:
        is_gpu_available = get_availability(GPU_NAME, True) or get_availability(GPU_NAME, False)
        print(str(is_gpu_available) + " " + GPU_NAME)
        if is_gpu_available: break

while True:
    sound_alarm()