Rem experimentname=str(K)+"_"+str(intermediate_dim)+"_" + str(batch_size)+"_" + str(maxiter)

python dgmmmiyawaki.py 18 128 20 5000
python fid.py 18 128 20 5000
