#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.cuh"
#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

// 1st part - heat method
// setup LC & A matrix
// solve using chol like the template implementation to get u


template <typename T, uint32_t blockThreads>
__global__ static void setup_LC_matrix(const Context            context,
                                       const VertexAttribute<T> coords,
                                       DenseMatrix<T>           LC_Mat,
                                       const bool use_uniform_laplace)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& p_id, const VertexIterator& iter) {
        T sum_cot_weights = 0;
        for (uint32_t v = 0; v < iter.size(); ++v) {
            VertexHandle r_id = iter[v];
            VertexHandle s_id = (v == iter.size() - 1) ? iter[0] : iter[v + 1];

            T cot_alpha = edge_cotan_weight(p_id, r_id, s_id, coords);
            T cot_beta  = edge_cotan_weight(r_id, p_id, s_id, coords);
            T weight    = cot_alpha + cot_beta;

            sum_cot_weights += weight;
            LC_Mat(p_id, r_id) = weight;
        }
        LC_Mat(p_id, p_id) = -sum_cot_weights;
    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(
        block,
        shrd_alloc,
        init_lambda,
        [](VertexHandle) { return true; },
        !use_uniform_laplace);
}


template <typename T, uint32_t blockThreads>
__global__ static void setup_A_matrix(const Context            context,
                                      const VertexAttribute<T> coords,
                                      SparseMatrix<T>          A_mat,
                                      const bool use_uniform_laplace,
                                      const T    time_step)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& p_id, const VertexIterator& iter) {
        T area_sum = 0;
        for (uint32_t v = 0; v < iter.size(); ++v) {
            VertexHandle r_id = iter[v];
            VertexHandle s_id = (v == iter.size() - 1) ? iter[0] : iter[v + 1];

            T tri_area = partial_voronoi_area(p_id, r_id, s_id, coords);
            area_sum += tri_area;
        }
        T vertex_area     = area_sum / 3.0;
        A_mat(p_id, p_id) = vertex_area;
    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VF>(
        block,
        shrd_alloc,
        init_lambda,
        [](VertexHandle) { return true; },
        !use_uniform_laplace);
}


template <typename T>
void solve_using_chol(rxmesh::RXMeshStatic& rx)
{
    constexpr uint32_t blockThreads = 256;
    uint32_t           num_vertices = rx.get_num_vertices();

    auto coords = rx.get_input_vertex_coordinates();

    SparseMatrix<T> A_mat(rx);
    DenseMatrix<T>  LC_Mat(rx, num_vertices, 3);

    // Setup A matrix
    LaunchBox<blockThreads> launch_box_A;
    rx.prepare_launch_box({Op::VF},
                          launch_box_A,
                          (void*)setup_A_matrix<T, blockThreads>,
                          !Arg.use_uniform_laplace);
    setup_A_matrix<T, blockThreads>
        <<<launch_box_A.blocks,
           launch_box_A.num_threads,
           launch_box_A.smem_bytes_dyn>>>(rx.get_context(),
                                          *coords,
                                          A_mat,
                                          Arg.use_uniform_laplace,
                                          Arg.time_step);
    CUDA_ERROR(cudaDeviceSynchronize());

    // Setup LC matrix
    LaunchBox<blockThreads> launch_box_LC;
    rx.prepare_launch_box({Op::VV},
                          launch_box_LC,
                          (void*)setup_LC_matrix<T, blockThreads>,
                          !Arg.use_uniform_laplace);
    setup_LC_matrix<T, blockThreads><<<launch_box_LC.blocks,
                                       launch_box_LC.num_threads,
                                       launch_box_LC.smem_bytes_dyn>>>(
        rx.get_context(), *coords, LC_Mat, Arg.use_uniform_laplace);
    CUDA_ERROR(cudaDeviceSynchronize());

    A_mat.pre_solve(PermuteMethod::NSTDIS);
    A_mat.solve(LC_Mat, *coords, Solver::CHOL);

    coords->move(rxmesh::DEVICE, rxmesh::HOST);

    rx.get_polyscope_mesh()->updateVertexPositions(*coords);
    polyscope::show();

    LC_Mat.release();
    A_mat.release();
}