import torch

def E_inc(d, k_0, k, x):
    # Compute dot product in a batched manner
    dot_product = torch.sum(k * x, dim=1, keepdim=True)   # Sum along dimension 1 and keep dimension for broadcasting
    return d * torch.exp(-1j * k_0 * dot_product)

# Green function for Helmholtz equation
def Green_function(k_0, x, s):
    diff = x - s
    r = torch.norm(diff, dim=-1)  # Computes the norm along the last dimension
    G = torch.exp(-1j* k_0 * r) / (4.0 * torch.pi * r)
    return G

# Derivatives of Green function (on x1,x2,x3,s1,s2,s3):


def Derivative_Green_function_x1(k_0,x,s):
    r = torch.sqrt((x[...,0]-s[...,0])**2+(x[...,1]-s[...,1])**2+(x[...,2]-s[...,2])**2)
    gx = -(1j*k_0*r+1)*(x[...,0]-s[...,0])*torch.exp(-1j*k_0*r)/(4*torch.pi*r**3)
    return gx

def Derivative_Green_function_x2(k_0,x,s):
    r = torch.sqrt((x[...,0]-s[...,0])**2+(x[...,1]-s[...,1])**2+(x[...,2]-s[...,2])**2)
    gx = -(1j*k_0*r+1)*(x[...,1]-s[...,1])*torch.exp(-1j*k_0*r)/(4*torch.pi*r**3)
    return gx
def Derivative_Green_function_x3(k_0,x,s):
    r = torch.sqrt((x[...,0]-s[...,0])**2+(x[...,1]-s[...,1])**2+(x[...,2]-s[...,2])**2)
    gx = -(1j*k_0*r+1)*(x[...,2]-s[...,2])*torch.exp(-1j*k_0*r)/(4*torch.pi*r**3)
    return gx

def Derivative_Green_function_s1(k_0,x,s):
    r = torch.sqrt((x[...,0]-s[...,0])**2+(x[...,1]-s[...,1])**2+(x[...,2]-s[...,2])**2)
    gs = -(1j*k_0*r+1)*(s[...,0]-x[...,0])*torch.exp(-1j*k_0*r)/(4*torch.pi*r**3)
    return gs

def Derivative_Green_function_s2(k_0,x,s):
    r = torch.sqrt((x[...,0]-s[...,0])**2+(x[...,1]-s[...,1])**2+(x[...,2]-s[...,2])**2)
    gs = -(1j*k_0*r+1)*(s[...,1]-x[...,1])*torch.exp(-1j*k_0*r)/(4*torch.pi*r**3)
    return gs

def Derivative_Green_function_s3(k_0,x,s):
    r = torch.sqrt((x[...,0]-s[...,0])**2+(x[...,1]-s[...,1])**2+(x[...,2]-s[...,2])**2)
    gs = -(1j*k_0*r+1)*(s[...,2]-x[...,2])*torch.exp(-1j*k_0*r)/(4*torch.pi*r**3)
    return gs

def Derivative_Green_function(k_0,x,s,l):
    r = torch.sqrt((x[...,0]-s[...,0])**2+(x[...,1]-s[...,1])**2+(x[...,2]-s[...,2])**2)
    gs = torch.exp(-1j*k_0*r)*(1+1j*k_0*r)*(l[...,0]*(x[...,0]-s[...,0])+l[...,1]*(x[...,1]-s[...,1])+l[...,2]*(x[...,2]-s[...,2]))/(4*torch.pi*(r**3))
    return gs

def Second_Derivative_Green_function(k_0,x,s,n,l):
    r = torch.sqrt((x[...,0]-s[...,0])**2+(x[...,1]-s[...,1])**2+(x[...,2]-s[...,2])**2)
    g = torch.exp(-1j*k_0*r)/(4*torch.pi*r**3) * (((1+1j*k_0*r)*((l[...,0]*n[...,0])+(l[...,1]*n[...,1])+(l[...,2]*n[...,2])))
                                                  +(3*((-1j*k_0/r)-(1/(r**2)))+(k_0**2))*
                                                  (l[...,0]*(x[...,0]-s[...,0])+l[...,1]*(x[...,1]-s[...,1])+l[...,2]*(x[...,2]-s[...,2]))
                                                  *((n[...,0]*(x[...,0]-s[...,0]))+(n[...,1]*(x[...,1]-s[...,1]))+(n[...,2]*(x[...,2]-s[...,2]))))
    return g

#B_n
def B_n(k_0, x, s, nx, ns):
    return k_0 * (torch.cos(k_0 * (s[...,0] - x[...,0]))* nx[...,0] * ns[...,0]+
                  torch.cos(k_0 * (s[...,1] - x[...,1]))* nx[...,1] * ns[...,1]+
                  torch.cos(k_0 * (s[...,2] - x[...,2]))* nx[...,2] * ns[...,2])

#B_n

def C_n(k_0, x, s, nx):
    return (torch.sin(k_0 * (s[...,0] - x[...,0]))* nx[...,0] +
                  torch.sin(k_0 * (s[...,1] - x[...,1]))* nx[...,1] +
                  torch.sin(k_0 * (s[...,2] - x[...,2]))* nx[...,2])

# D_n
def D_n(k_0,n,nl):
    return (n[0,...,0]*nl[0,...,0]+n[0,...,1]*nl[0,...,1]+n[0,...,2]*nl[0,...,2])*k_0

def G_mm(areas, k_0, triangle_centers, normal_vectors, N,device):
    mask = torch.eye(N, device=device)== 0
    repeated_x = torch.stack([triangle_centers[i].expand(N, -1).to(torch.complex128) for i in range(N)])
    s = triangle_centers.to(torch.complex128)
    repeated_s = s.unsqueeze(0).repeat(N, 1, 1)
    repeated_nx = torch.stack([normal_vectors[i].expand(N, -1).to(torch.complex128) for i in range(N)])
    n = normal_vectors.to(torch.complex128)
    repeated_ns = n.unsqueeze(0).repeat(N, 1, 1)
    B = B_n(k_0, repeated_x, repeated_s, repeated_nx, repeated_ns)
    C = C_n(k_0, repeated_x, repeated_s, repeated_nx)
    G= torch.where(mask, Green_function(k_0, repeated_x, repeated_s), torch.zeros((N, N), device=device)).to(dtype=torch.complex128)
    D_G_s1 = torch.where(mask, Derivative_Green_function_s1(k_0, repeated_x, repeated_s), torch.zeros((N, N), device=device)).to(dtype=torch.complex128)
    D_G_s2 = torch.where(mask, Derivative_Green_function_s2(k_0, repeated_x, repeated_s), torch.zeros((N, N), device=device)).to(dtype=torch.complex128)
    D_G_s3 = torch.where(mask, Derivative_Green_function_s3(k_0, repeated_x, repeated_s), torch.zeros((N, N), device=device)).to(dtype=torch.complex128)
    ns1=repeated_ns[0,...,0].unsqueeze(0).repeat(N, 1, 1).squeeze(1)
    ns2=repeated_ns[0,...,1].unsqueeze(0).repeat(N, 1, 1).squeeze(1)
    ns3=repeated_ns[0,...,2].unsqueeze(0).repeat(N, 1, 1).squeeze(1)
    area_expanded = areas.unsqueeze(0).repeat(N, 1, 1).squeeze(1)
    EQ = ((G * B) - (((D_G_s1 * ns1) + (D_G_s2 * ns2) + (D_G_s3 * ns3)) * C)) * area_expanded
    total = torch.sum(EQ, dim=1)
    del EQ
    F= -total/(k_0*areas)
    return F

def G_mm_derivative(nl, areas, k_0, triangle_centers, normal_vectors, N,device):
    mask = torch.eye(N, device=device) == 0
    repeated_x = torch.stack([triangle_centers[i].expand(N, -1).to(torch.complex128) for i in range(N)])
    s = triangle_centers.to(torch.complex128)
    repeated_s = s.unsqueeze(0).repeat(N, 1, 1)
    repeated_nx = torch.stack([normal_vectors[i].expand(N, -1).to(torch.complex128) for i in range(N)])
    n = normal_vectors.to(torch.complex128)
    repeated_ns = n.unsqueeze(0).repeat(N, 1, 1)
    B= B_n(k_0, repeated_x, repeated_s, repeated_nx, repeated_ns)
    C = C_n(k_0, repeated_x, repeated_s, repeated_nx)
    area_expanded = areas.unsqueeze(0).repeat(N, 1, 1).squeeze(1)
    repeated_nl = nl.unsqueeze(0).repeat(N, 1, 1)
    D_G = torch.where(mask, Derivative_Green_function(k_0, repeated_x, repeated_s,repeated_nl),torch.zeros((N, N), device=device)).to(dtype=torch.complex128)
    DD_G = torch.where(mask, Second_Derivative_Green_function(k_0, repeated_x, repeated_s, repeated_ns,repeated_nl), torch.zeros((N, N), device=device)).to(dtype=torch.complex128)
    D = D_n(k_0,repeated_ns,repeated_nl)/torch.pi/4
    EQ = ((-D_G * B) - (DD_G * C)) * area_expanded
    total = torch.sum(EQ, dim=1)
    F= -(total - 0.5*D) / (k_0 * areas)
    return F

def Electric_field(x,triangle_centers,alpha1,alpha2,alpha3,k_0,N):
    x_vector = x.unsqueeze(2).expand(-1, -1, N, -1)
    T = torch.cat([triangle_centers], dim=0)
    T_repeated = T.unsqueeze(0).unsqueeze(0).repeat(len(x), len(x[0]), 1, 1)
    R1 = alpha1 * Green_function(k_0, x_vector, T_repeated)
    R2 = alpha2 * Green_function(k_0, x_vector, T_repeated)
    R3 = alpha3 * Green_function(k_0, x_vector, T_repeated)
    R4 = Green_function(k_0, x_vector, T_repeated)
    E_x = torch.sum(R1, dim=2)
    E_y = torch.sum(R2, dim=2)
    E_z = torch.sum(R3, dim=2)
    G_f = torch.sum(R4, dim=2)
    return E_x, E_y, E_z,  G_f