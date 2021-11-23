def positional_encoding(pos,model_size):
  '''
  This function makes the positional embedding vector for every token.
  takes a position of a token in sentence and returns vector. It has 2 different equations,
  one for even positions and one for odd positions.
  
  Inputs:
  pos: position of the token.
  model_size: embedding layer dimension our model will use.
  
  output:
  POS: vector that contains encoding for position.
  
  helpful links:
  https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
  
 '''
  POS = np.zeros((1,model_size)) # [0,0,0,....,0]
  for i in range(model_size):
    if i % 2 == 0: #even number
      POS[:,i] = np.sin(pos/10000 *(i / model_size)) # here we used [:,i] because we already have on row only
    else:
      POS[:,i] = np.cos(pos/10000 *( (i-1) / model_size)) # here we used i -1 because in equation for odd i takes position of the prevoius token
      # will add more explanations about i - 1
      
  return POS
  
  
