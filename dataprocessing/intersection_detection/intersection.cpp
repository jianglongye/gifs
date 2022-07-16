#include <fstream>
#include <iostream>
#include <list>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/IO/OFF_reader.h>
#include <CGAL/IO/OBJ_reader.h>

typedef CGAL::Simple_cartesian<double> K;
typedef K::FT FT;
typedef K::Ray_3 Ray;
typedef K::Segment_3 Segment;
typedef K::Line_3 Line;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
typedef CGAL::Surface_mesh<Point> Mesh;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Args error!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string mesh_path = argv[1];
    std::string bin_path = argv[2];
    std::string output_path = argv[3];

    std::ifstream fin = std::ifstream(mesh_path);
    std::vector<Point> points;
    std::vector<std::vector<std::size_t>> faces;
    CGAL::read_OFF(fin, points, faces);

    std::list<Triangle> triangles;
    for (std::vector<std::size_t> face : faces)
    {
        triangles.push_back(Triangle(points[face[0]], points[face[1]], points[face[2]]));
    }
    // std::cout << triangles.size() << std::endl;

    // constructs AABB tree
    Tree tree(triangles.begin(), triangles.end());

    float f;
    std::ofstream fout = std::ofstream(output_path, std::ios::binary);

    fin.close();
    fin.open(bin_path, std::ios::binary);
    std::vector<float> data;

    bool result;
    Segment segment_query;

    int i = 0;
    while (fin.read(reinterpret_cast<char *>(&f), sizeof(float)))
    {
        data.push_back(f);
        i++;
        if (i % 6 == 0)
        {
            // std::cout << data[0] << std::endl;
            segment_query = Segment(Point(data[0], data[1], data[2]), Point(data[3], data[4], data[5]));
            result = tree.do_intersect(segment_query);
            fout.write(reinterpret_cast<char *>(&result), sizeof(bool));
            data = std::vector<float>();
        }
    }

    fin.close();
    fout.close();
    return EXIT_SUCCESS;
}