from models import Generator, Discriminator
import torch

gen = Generator()

dis = Discriminator()


# output = gen(input)

# print(output.shape)

xi = torch.rand((3, 1, 208, 160))

h = torch.rand((3, 2))

a = torch.rand((3, 100))

gen_output = gen(xi, h, a)

print(gen_output.shape)


dis_output = dis(gen_output, h, a)
print(dis_output.shape)