import numpy as np
import pytest
import hypothesis.strategies as st
from hypothesis import given, assume, example
from TacSegmentViz import inverse_rotation, rotate_indices

rot = st.integers(min_value=0)
shape = st.tuples(st.integers(min_value=1), st.integers(min_value=1))

@st.composite
def coordinates_and_shape(draw, index=st.integers()):
    d1 =draw(st.integers(min_value=1, max_value=1024))
    d2 =draw(st.integers(min_value=1, max_value=1024))
    ind1 =draw(st.integers(min_value=1))
    ind2 =draw(st.integers(min_value=1))
    assume(ind1<d1)
    assume(ind2<d2)
    shape = (d1,d2)
    coordinates = (ind1,ind2)
    return [coordinates,shape]

@given(rot=rot,cns=coordinates_and_shape())
def test_nrot_inverse(rot,cns):
    '''
    since inverse is based on rotate indices testing one tests the other as well
    '''
    coordinates = cns[0]
    shape = cns[1]
    mat = np.random.rand(shape[0], shape[1])
    coord = rotate_indices(mat,rot,coordinates)
    rot_mat = np.rot90(mat,rot)
    res = inverse_rotation(rot_mat, rot, coord)
    assert mat[res] == rot_mat[coord] and res==coordinates

@given(rot=rot,cns=coordinates_and_shape())
@example(rot=0, cns=[(2,5),(512,512)])
def test_idempotence(rot,cns):
    #force rotations as multiples of four to test idempotence of the transformation. once again the inverse is defined on the original so testing inverse tests both
    coord = cns[0]
    shape = cns[1]
    mat = np.zeros(shape=shape)
    rot_mat = np.rot90(mat, rot)
    res = inverse_rotation(rot_mat, rot*4, coord)
    assert res==coord

@given(rot=rot,cns=coordinates_and_shape())
def test_inverse_is_inverse(rot,cns):
    coord = cns[0]
    shape = cns[1]
    mat = np.zeros(shape=shape)
    rot_mat = np.rot90(mat,rot)
    res = rotate_indices(mat, rot, coord)
    assert coord == inverse_rotation(rot_mat,rot,res)

@given(rot=rot,cns=coordinates_and_shape())
def test_original_is_inverse_of_inverse(rot,cns):
    coord = cns[0]
    shape = cns[1]
    #since i'm testing the inverse i generate a matrix i consider rotated and then rotate it backwards to obtain the original
    rotated_matrix = np.zeros(shape=shape)#np.asarray(np.random.randint(100,size=(dim1,dim2)))
    matrix_rotated_backwards = np.rot90(rotated_matrix,-rot) #Rot is defined positive so here i try to have it rotate backwards
    res = inverse_rotation(rotated_matrix, rot, coord)
    assert coord == rotate_indices(matrix_rotated_backwards,rot,res)

@given(rot=rot,cns=coordinates_and_shape())
def test_non_negative_output(rot,cns):
    coord = cns[0]
    shape = cns[1]
    mat = np.zeros(shape=shape)#np.asarray(np.random.randint(100,size=(dim1,dim2)))
    res1, res2 = inverse_rotation(mat, rot, coord)
    assert res1>=0 and res2>=0

@given(rot=rot,cns=coordinates_and_shape())
def test_still_within_bounds(rot,cns):
    coord = cns[0]
    shape = cns[1]
    #generate the rotated matrix with valid coordinates, rotate it bwrd to obtain the original and check if the results are within bounds
    rotated_matrix = np.zeros(shape=shape)#np.asarray(np.random.randint(100,size=(dim1,dim2)))
    matrix_rotated_backwards = np.rot90(rotated_matrix,-rot)
    res1, res2 = inverse_rotation(rotated_matrix, rot, coord)
    assert res1<matrix_rotated_backwards.shape[0] and res2<matrix_rotated_backwards.shape[1]
'''
@given(rot=st.integers(min_value=1))
def test_inductivity(rot):

    #Probably this just tests the inversion of the function

    #force rotations as multiples of four to test idempotence of the transformation. once again the inverse is defined on the original so testing inverse tests both
    mat = np.asarray(np.random.randint(100,size=(np.random.randint(1000),np.random.randint(1000))))
    rot_mat = np.rot90(mat, rot)
    coord = (np.random.randint(mat.shape[0]), np.random.randint(mat.shape[1]))
    res1 = rotate_indices(mat, rot-1, coord)
    assert coord==inverse_rotation(np.rot90(mat,rot-1), rot-1, res1)
'''
