# Couleurs ANSI
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_box(text, color):
	"""Print text inside a box whose width follows the text length."""
	inner = f"   {text}   "
	print(f"{color}╔{'═' * len(inner)}╗{RESET}")
	print(f"{color}║{inner}║{RESET}")
	print(f"{color}╚{'═' * len(inner)}╝{RESET}")

def load_theta(path="theta.txt"):
	"""Return (theta0, theta1) saved by train.py, or (0, 0) if no training has been done yet."""
	try:
		with open(path) as f:
			theta0 = float(f.readline())
			theta1 = float(f.readline())
		return theta0, theta1
	except (OSError, ValueError):
		return 0.0, 0.0

def estimate_price(mileage, theta0, theta1):
	"""estimatePrice(mileage) = theta0 + (theta1 * mileage)"""
	return theta0 + (theta1 * mileage)


print("\033c", end="")

theta0, theta1 = load_theta()
i = 0

try:
	while (True) :
		try :
			if i == 0 :
				print_box("Input (km)", f"{CYAN}{BOLD}")
				if theta0 == 0 and theta1 == 0 :
					print(f"{YELLOW}⚠️  No training found (theta.txt): prediction will be 0{RESET}")
			prompt = input(f"➡️  {YELLOW}Enter mileage (in km) or {RESET}{RED}EXIT {RESET}{YELLOW}for quit : {RESET}")
			if (prompt == "EXIT") :
				print("\033c", end="")
				break
			mileage = float(prompt)
			if not (0 <= mileage < float("inf")) :  # rejects negative, nan and inf
				print("\033c", end="")
				i = 1
				print_box("Please enter a positive number", RED)
				print()
				continue
			price = estimate_price(mileage, theta0, theta1)
			if price < 0 :
				print("\033c", end="")
				i = 1
				print_box("Mileage too high", RED)
				print()
				continue
			print("\033c", end="")
			i = 0
			print_box(f"Estimated price: {price:.2f} euros", GREEN)
		except ValueError:
			i = 1
			print("\033c", end="")
			print_box("Please enter a valid number", RED)
			print()

except (KeyboardInterrupt, EOFError):
	print("\033c", end="")
