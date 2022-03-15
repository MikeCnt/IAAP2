

def desGradientAdjustmentMSE(weights, b, db, u, yEstimated):

	lastElement = len(db[0]) - 1
	q = len(db)

	for i in range(len(db)):
		sum = 0

		for j in range(len(db[i])):

			for k in range(q):

				sum += (yEstimated - db[k][lastElement]) * db[i][j]

			weights[j] = weights[j] - (u/q) * sum

	
	sum  = 0

	for i in range(q):
		sum += (yEstimated - db[i][lastElement])

	b = b - (u/q) * sum

	return weights, b