import subprocess
import re
import time

# Loop infinito para atualizar as informações
while True:
    # Limpar a tela (funciona em ambientes UNIX/Linux)
    print("\033[H\033[J", end="")

    # Execute o comando nvidia-smi
    nvidia_smi_output = subprocess.check_output("nvidia-smi", encoding='UTF-8')

    # Expressões regulares para capturar as informações
    gpu_info_pattern = re.compile(
        r"\|\s+(\d+)\s+NVIDIA ([\w\s]+)\s+Off.*?\|\s+(\d+)%\s+(\d+C)\s+.*?\s+(\d+W) / (\d+W)\s+\|\s+(\d+MiB) / (\d+MiB)\s+\|\s+(\d+)%.*?\|",
        re.DOTALL)

    # Encontrar todas as correspondências
    matches = gpu_info_pattern.findall(nvidia_smi_output)

    # Processar e imprimir as informações capturadas
    for match in matches:
        gpu_id, gpu_name, gpu_fan, gpu_temp, gpu_power_usage, gpu_power_total, mem_usage, mem_total, gpu_util = match
        print(f"GPU ID: {gpu_id}")
        print(f"GPU Name: {gpu_name.strip()}")
        print(f"Temperature: {gpu_temp}")
        print(f"Power Usage/Total: {gpu_power_usage}/{gpu_power_total}")
        print(f"Memory Usage/Total: {mem_usage}/{mem_total}")
        print(f"GPU Utilization: {gpu_util}%")
        print("-" * 40)

    # Imprimir a quantidade total de GPUs
    print(f"Total GPUs: {len(matches)}")

    # Esperar por 1.5 segundos antes da próxima atualização
    time.sleep(1.5)
