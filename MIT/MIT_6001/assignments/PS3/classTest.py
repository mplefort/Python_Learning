class Rabbit(object):
	tag = 0

	def __init__(self, age, name):
		self.age = age
		self.name = name
		self.rid = Rabbit.tag
		Rabbit.tag += 1



r1 = Rabbit(10,"fluffy")
r2 = Rabbit(3,"squeky")

print("rabbit 1: " + str(r1.rid))
print("rabbit 2: " + str(r2.rid))

print(r1.tag)
print(Rabbit.tag)