def feature(a: int, b: int):
	product = 0
	for i in range(b):
		product += a

	return product

if __name__ == "__main__":
	print(feature(3, 4))
