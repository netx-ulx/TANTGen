def check_detection(logfile, keywords=["telnet", "syn", "scan", "mirai"]):
    with open(logfile, "r") as f:
        content = f.read().lower()
        return any(kw in content for kw in keywords)

path = "ids_output/Mirai_adv/logs.txt"
if check_detection(path):
    print("➡️  Aversarial attack DETECTED!")
else:
    print("✅ Aversarial attack NOT DETECTED!")