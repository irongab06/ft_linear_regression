import numpy as np
import os

# Couleurs ANSI
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

print("\033c", end="")

try:
	i = 0
	theta = np.loadtxt("theta.txt")
	theta1, theta0 = float(theta[0]), float(theta[1])
	while (True) :
		try :
			if i == 0 :
				print(f"{CYAN}{BOLD}╔══════════════════════════════╗{RESET}")
				print(f"{CYAN}{BOLD}║          Input (km)          ║{RESET}")
				print(f"{CYAN}{BOLD}╚══════════════════════════════╝{RESET}")
			prompt = input(f"➡️  {YELLOW}Enter mileage (in km) or {RESET}{RED}EXIT {RESET}{YELLOW}for quit : {RESET}")
			if (prompt == "EXIT") :
				print("\033c", end="")
				break
			mileage = float(prompt)
			if mileage < 0 or mileage != mileage :
				print("\033c", end="")
				i = 1
				print(f"{RED}╔══════════════════════════════════════╗{RESET}")
				print(f"{RED}║   Please enter a positive number     ║{RESET}")
				print(f"{RED}╚══════════════════════════════════════╝{RESET}\n")
				continue
			price = theta0 + theta1 * (mileage / 10000.0)
			if price < 0 :
				print("\033c", end="")
				i = 1
				print(f"{RED}╔════════════════════════╗{RESET}")
				print(f"{RED}║   Mileage too high     ║{RESET}")
				print(f"{RED}╚════════════════════════╝{RESET}\n")
				continue
			print("\033c", end="")
			i = 0
			print(f"{GREEN}╔══════════════════════════════════════════╗{RESET}")
			print(f"{GREEN}║      Estimated price: {price:.2f} euros      ║{RESET}")
			print(f"{GREEN}╚══════════════════════════════════════════╝{RESET}")
		except ValueError:
			i = 1
			print("\033c", end="")
			print(f"{RED}╔══════════════════════════════════════╗{RESET}")
			print(f"{RED}║     Please enter a valid number      ║{RESET}")
			print(f"{RED}╚══════════════════════════════════════╝{RESET}\n")

except Exception as e :
	print("\033c", end="")
	print(f"{RED}╔════════════════════════════════════════════════════╗{RESET}")
	print(f"{RED}║    Could not load theta.txt: {e}  ║{RESET}")
	print(f"{RED}╚════════════════════════════════════════════════════╝{RESET}")
