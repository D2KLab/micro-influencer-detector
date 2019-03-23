fall = open("user_list.csv", "r")
all_users = fall.read().split(",")
fall.close()

falready = open("out.txt", "r")
already = falready.read().split("\n")
falready.close()

remained = set(all_users) - set(already)
fdiff = open("diff.csv", "w")
for user in remained:
	fdiff.write(user+",")
fdiff.close() 