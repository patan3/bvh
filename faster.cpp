#include "precomp.h"
#include "faster.h"
#include <vector>
#include <cmath> // abs()
using namespace std;

// Global counters
unsigned int global_traversal_steps = 0;
unsigned int global_intersect_tests = 0;

// THIS SOURCE FILE:
// Code for the article "How to Build a BVH", part 2: faster rays.
// This version improves ray traversal speed using ordered traversal
// and better split plane orientation / positions based on the SAH
// (Surface Area Heuristic).
// Feel free to copy this code to your own framework. Absolutely no
// rights are reserved. No responsibility is accepted either.
// For updates, follow me on twitter: @j_bikker.

TheApp* CreateApp() { return new FasterRaysApp(); }

// enable the use of SSE in the AABB intersection function
#define USE_SSE


#define N	5000
// triangle count
// #define MAX_TRIS 25000       // Maximum space we reserve
//int N;

// Settings
#define GRID_SIZE 40 
// We use a 40x40x40 grid. 
// 40^3 = 64,000 cells. This is a good balance for 20k triangles.

// forward declarations
void Subdivide( uint nodeIdx );
void UpdateNodeBounds( uint nodeIdx );

// minimal structs
struct Tri { float3 vertex0, vertex1, vertex2; float3 centroid; };
struct BVHNode
{
	union { struct { float3 aabbMin; uint leftFirst; }; __m128 aabbMin4; };
	union { struct { float3 aabbMax; uint triCount; }; __m128 aabbMax4; };
	bool isLeaf() { return triCount > 0; }
};
struct aabb
{
	float3 bmin = 1e30f, bmax = -1e30f;
	void grow( float3 p ) { bmin = fminf( bmin, p ); bmax = fmaxf( bmax, p ); }
	float area()
	{
		float3 e = bmax - bmin; // box extent
		return e.x * e.y + e.y * e.z + e.z * e.x;
	}
};
__declspec(align(64)) struct Ray
{
	Ray() { O4 = D4 = rD4 = _mm_set1_ps( 1 ); }
	union { struct { float3 O; float dummy1; }; __m128 O4; };
	union { struct { float3 D; float dummy2; }; __m128 D4; };
	union { struct { float3 rD; float dummy3; }; __m128 rD4; };
	float t = 1e30f;
};
struct Grid {
	vector<int> cells[GRID_SIZE * GRID_SIZE * GRID_SIZE];
	float3 bmin, bmax;
	float3 cellSize;

	// Helper to map 3D (x,y,z) to 1D index
	int GetIdx(int x, int y, int z) {
		// Clamp indices to stay inside the array
		if (x < 0) x = 0; if (x >= GRID_SIZE) x = GRID_SIZE - 1;
		if (y < 0) y = 0; if (y >= GRID_SIZE) y = GRID_SIZE - 1;
		if (z < 0) z = 0; if (z >= GRID_SIZE) z = GRID_SIZE - 1;
		return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
	}
} grid;

// application data
Tri tri[N];
uint triIdx[N];
BVHNode* bvhNode = 0;
uint rootNodeIdx = 0, nodesUsed = 2;

// functions

void IntersectTri( Ray& ray, const Tri& tri )
{
	const float3 edge1 = tri.vertex1 - tri.vertex0;
	const float3 edge2 = tri.vertex2 - tri.vertex0;
	const float3 h = cross( ray.D, edge2 );
	const float a = dot( edge1, h );
	if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
	const float f = 1 / a;
	const float3 s = ray.O - tri.vertex0;
	const float u = f * dot( s, h );
	if (u < 0 || u > 1) return;
	const float3 q = cross( s, edge1 );
	const float v = f * dot( ray.D, q );
	if (v < 0 || u + v > 1) return;
	const float t = f * dot( edge2, q );
	if (t > 0.0001f) ray.t = min( ray.t, t );
}

inline float IntersectAABB( const Ray& ray, const float3 bmin, const float3 bmax )
{
	float tx1 = (bmin.x - ray.O.x) * ray.rD.x, tx2 = (bmax.x - ray.O.x) * ray.rD.x;
	float tmin = min( tx1, tx2 ), tmax = max( tx1, tx2 );
	float ty1 = (bmin.y - ray.O.y) * ray.rD.y, ty2 = (bmax.y - ray.O.y) * ray.rD.y;
	tmin = max( tmin, min( ty1, ty2 ) ), tmax = min( tmax, max( ty1, ty2 ) );
	float tz1 = (bmin.z - ray.O.z) * ray.rD.z, tz2 = (bmax.z - ray.O.z) * ray.rD.z;
	tmin = max( tmin, min( tz1, tz2 ) ), tmax = min( tmax, max( tz1, tz2 ) );
	if (tmax >= tmin && tmin < ray.t && tmax > 0) return tmin; else return 1e30f;
}

float IntersectAABB_SSE( const Ray& ray, const __m128& bmin4, const __m128& bmax4 )
{
	static __m128 mask4 = _mm_cmpeq_ps( _mm_setzero_ps(), _mm_set_ps( 1, 0, 0, 0 ) );
	__m128 t1 = _mm_mul_ps( _mm_sub_ps( _mm_and_ps( bmin4, mask4 ), ray.O4 ), ray.rD4 );
	__m128 t2 = _mm_mul_ps( _mm_sub_ps( _mm_and_ps( bmax4, mask4 ), ray.O4 ), ray.rD4 );
	__m128 vmax4 = _mm_max_ps( t1, t2 ), vmin4 = _mm_min_ps( t1, t2 );
	float tmax = min( vmax4.m128_f32[0], min( vmax4.m128_f32[1], vmax4.m128_f32[2] ) );
	float tmin = max( vmin4.m128_f32[0], max( vmin4.m128_f32[1], vmin4.m128_f32[2] ) );
	if (tmax >= tmin && tmin < ray.t && tmax > 0) return tmin; else return 1e30f;
}

void IntersectGrid(Ray& ray)
{
	// 1. Intersect Ray with the Grid Bounding Box
	float tMin = IntersectAABB(ray, grid.bmin, grid.bmax);
	if (tMin == 1e30f) return; // Miss

	// If we are inside the grid, start at 0. Otherwise start at entry point.
	float tCurrent = (tMin > 0) ? tMin : 0;

	// 2. Initialize Position in Grid
	float3 startPos = ray.O + ray.D * tCurrent;
	// Calculate integer coordinates (x,y,z)
	int X = (startPos.x - grid.bmin.x) / grid.cellSize.x;
	int Y = (startPos.y - grid.bmin.y) / grid.cellSize.y;
	int Z = (startPos.z - grid.bmin.z) / grid.cellSize.z;

	// Clamp to be safe
	X = max(0, min(X, GRID_SIZE - 1));
	Y = max(0, min(Y, GRID_SIZE - 1));
	Z = max(0, min(Z, GRID_SIZE - 1));

	// 3. Setup Amanatides & Woo "Step" variables
	// "How far do we travel to cross the next X line?"
	int stepX = (ray.D.x > 0) ? 1 : -1;
	int stepY = (ray.D.y > 0) ? 1 : -1;
	int stepZ = (ray.D.z > 0) ? 1 : -1;

	float tDeltaX = abs(grid.cellSize.x / ray.D.x);
	float tDeltaY = abs(grid.cellSize.y / ray.D.y);
	float tDeltaZ = abs(grid.cellSize.z / ray.D.z);

	// Calculate distance to FIRST boundary
	float nextXBound = grid.bmin.x + (X + (stepX > 0 ? 1 : 0)) * grid.cellSize.x;
	float nextYBound = grid.bmin.y + (Y + (stepY > 0 ? 1 : 0)) * grid.cellSize.y;
	float nextZBound = grid.bmin.z + (Z + (stepZ > 0 ? 1 : 0)) * grid.cellSize.z;

	float tMaxX = (nextXBound - ray.O.x) / ray.D.x;
	float tMaxY = (nextYBound - ray.O.y) / ray.D.y;
	float tMaxZ = (nextZBound - ray.O.z) / ray.D.z;

	// 4. Walk the Grid
	while (true)
	{
		global_traversal_steps++;

		// Check triangles in current cell
		int idx = grid.GetIdx(X, Y, Z);
		for (int i : grid.cells[idx]) {
			global_intersect_tests++;
			IntersectTri(ray, tri[i]);
		}

		// Optimisation: Stop if we found a hit inside this cell
		if (ray.t < tMaxX && ray.t < tMaxY && ray.t < tMaxZ) return;

		// Step to next cell (The core algorithm)
		if (tMaxX < tMaxY) {
			if (tMaxX < tMaxZ) {
				X += stepX; tMaxX += tDeltaX;
			}
			else {
				Z += stepZ; tMaxZ += tDeltaZ;
			}
		}
		else {
			if (tMaxY < tMaxZ) {
				Y += stepY; tMaxY += tDeltaY;
			}
			else {
				Z += stepZ; tMaxZ += tDeltaZ;
			}
		}

		// Check if we left the grid
		if (X < 0 || X >= GRID_SIZE || Y < 0 || Y >= GRID_SIZE || Z < 0 || Z >= GRID_SIZE) break;
	}
}

void IntersectBVH( Ray& ray )
{
	BVHNode* node = &bvhNode[rootNodeIdx], * stack[64];
	uint stackPtr = 0;
	while (1)
	{
		// TRAVERSAL STEPS
		// Every time the loop runs, we are visiting a new Node (a new Box).
		global_traversal_steps++;
		if (node->isLeaf())
		{
			// If leaf node, we check the triangles inside it.
			for (uint i = 0; i < node->triCount; i++) {
				// INTERSECTION TESTS
				// Check a specific triangle.
				global_intersect_tests++;

				IntersectTri(ray, tri[triIdx[node->leftFirst + i]]);
			}
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		BVHNode* child1 = &bvhNode[node->leftFirst];
		BVHNode* child2 = &bvhNode[node->leftFirst + 1];
	#ifdef USE_SSE
		float dist1 = IntersectAABB_SSE( ray, child1->aabbMin4, child1->aabbMax4 );
		float dist2 = IntersectAABB_SSE( ray, child2->aabbMin4, child2->aabbMax4 );
	#else
		float dist1 = IntersectAABB( ray, child1->aabbMin, child1->aabbMax );
		float dist2 = IntersectAABB( ray, child2->aabbMin, child2->aabbMax );
	#endif
		if (dist1 > dist2) { swap( dist1, dist2 ); swap( child1, child2 ); }
		if (dist1 == 1e30f)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			node = child1;
			if (dist2 != 1e30f) stack[stackPtr++] = child2;
		}
	}
}

void BuildBVH()
{
	// create the BVH node pool
	bvhNode = (BVHNode*)_aligned_malloc( sizeof( BVHNode ) * N * 2, 64 );
	// populate triangle index array
	for (int i = 0; i < N; i++) triIdx[i] = i;
	// calculate triangle centroids for partitioning
	for (int i = 0; i < N; i++)
		tri[i].centroid = (tri[i].vertex0 + tri[i].vertex1 + tri[i].vertex2) * 0.3333f;
	// assign all triangles to root node
	BVHNode& root = bvhNode[rootNodeIdx];
	root.leftFirst = 0, root.triCount = N;
	UpdateNodeBounds( rootNodeIdx );
	// subdivide recursively
	Timer t;
	Subdivide( rootNodeIdx );
	printf( "BVH (%i nodes) constructed in %.2fms.\n", nodesUsed, t.elapsed() * 1000 );
}

void UpdateNodeBounds( uint nodeIdx )
{
	BVHNode& node = bvhNode[nodeIdx];
	node.aabbMin = float3( 1e30f );
	node.aabbMax = float3( -1e30f );
	for (uint first = node.leftFirst, i = 0; i < node.triCount; i++)
	{
		uint leafTriIdx = triIdx[first + i];
		Tri& leafTri = tri[leafTriIdx];
		node.aabbMin = fminf( node.aabbMin, leafTri.vertex0 );
		node.aabbMin = fminf( node.aabbMin, leafTri.vertex1 );
		node.aabbMin = fminf( node.aabbMin, leafTri.vertex2 );
		node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex0 );
		node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex1 );
		node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex2 );
	}
}

float EvaluateSAH( BVHNode& node, int axis, float pos )
{
	// determine triangle counts and bounds for this split candidate
	aabb leftBox, rightBox;
	int leftCount = 0, rightCount = 0;
	for (uint i = 0; i < node.triCount; i++)
	{
		Tri& triangle = tri[triIdx[node.leftFirst + i]];
		if (triangle.centroid[axis] < pos)
		{
			leftCount++;
			leftBox.grow( triangle.vertex0 );
			leftBox.grow( triangle.vertex1 );
			leftBox.grow( triangle.vertex2 );
		}
		else
		{
			rightCount++;
			rightBox.grow( triangle.vertex0 );
			rightBox.grow( triangle.vertex1 );
			rightBox.grow( triangle.vertex2 );
		}
	}
	float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
	return cost > 0 ? cost : 1e30f;
}

void BuildGrid() {
	printf("Building Grid from GitHub algorithm...\n");

	// 1. Calculate Scene Bounds
	grid.bmin = float3(1e30f); grid.bmax = float3(-1e30f);
	for (int i = 0; i < N; i++) {
		grid.bmin = fminf(grid.bmin, tri[i].vertex0);
		grid.bmin = fminf(grid.bmin, tri[i].vertex1);
		grid.bmin = fminf(grid.bmin, tri[i].vertex2);
		grid.bmax = fmaxf(grid.bmax, tri[i].vertex0);
		grid.bmax = fmaxf(grid.bmax, tri[i].vertex1);
		grid.bmax = fmaxf(grid.bmax, tri[i].vertex2);
	}
	// Add small buffer
	grid.bmin -= 0.01f; grid.bmax += 0.01f;

	// 2. Determine Cell Size
	float3 size = grid.bmax - grid.bmin;
	grid.cellSize = float3(size.x / GRID_SIZE, size.y / GRID_SIZE, size.z / GRID_SIZE);

	// 3. Insert Triangles (Binning)
	for (int i = 0; i < N; i++) {
		// Compute the range of cells this triangle touches
		float3 triMin = fminf(tri[i].vertex0, fminf(tri[i].vertex1, tri[i].vertex2));
		float3 triMax = fmaxf(tri[i].vertex0, fmaxf(tri[i].vertex1, tri[i].vertex2));

		int minX = (triMin.x - grid.bmin.x) / grid.cellSize.x;
		int minY = (triMin.y - grid.bmin.y) / grid.cellSize.y;
		int minZ = (triMin.z - grid.bmin.z) / grid.cellSize.z;
		int maxX = (triMax.x - grid.bmin.x) / grid.cellSize.x;
		int maxY = (triMax.y - grid.bmin.y) / grid.cellSize.y;
		int maxZ = (triMax.z - grid.bmin.z) / grid.cellSize.z;

		for (int x = minX; x <= maxX; x++)
			for (int y = minY; y <= maxY; y++)
				for (int z = minZ; z <= maxZ; z++)
					grid.cells[grid.GetIdx(x, y, z)].push_back(i);
	}
}

void Subdivide( uint nodeIdx )
{
	// terminate recursion
	BVHNode& node = bvhNode[nodeIdx];
	// determine split axis using SAH
	int bestAxis = -1;
	float bestPos = 0, bestCost = 1e30f;
	for (int axis = 0; axis < 3; axis++) for (uint i = 0; i < node.triCount; i++)
	{
		Tri& triangle = tri[triIdx[node.leftFirst + i]];
		float candidatePos = triangle.centroid[axis];
		float cost = EvaluateSAH( node, axis, candidatePos );
		if (cost < bestCost)
			bestPos = candidatePos, bestAxis = axis, bestCost = cost;
	}
	int axis = bestAxis;
	float splitPos = bestPos;
	float3 e = node.aabbMax - node.aabbMin; // extent of parent
	float parentArea = e.x * e.y + e.y * e.z + e.z * e.x;
	float parentCost = node.triCount * parentArea;
	if (bestCost >= parentCost) return;
	// in-place partition
	int i = node.leftFirst;
	int j = i + node.triCount - 1;
	while (i <= j)
	{
		if (tri[triIdx[i]].centroid[axis] < splitPos)
			i++;
		else
			swap( triIdx[i], triIdx[j--] );
	}
	// abort split if one of the sides is empty
	int leftCount = i - node.leftFirst;
	if (leftCount == 0 || leftCount == node.triCount) return;
	// create child nodes
	int leftChildIdx = nodesUsed++;
	int rightChildIdx = nodesUsed++;
	bvhNode[leftChildIdx].leftFirst = node.leftFirst;
	bvhNode[leftChildIdx].triCount = leftCount;
	bvhNode[rightChildIdx].leftFirst = i;
	bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
	node.leftFirst = leftChildIdx;
	node.triCount = 0;
	UpdateNodeBounds( leftChildIdx );
	UpdateNodeBounds( rightChildIdx );
	// recurse
	Subdivide( leftChildIdx );
	Subdivide( rightChildIdx );
}

void FasterRaysApp::Init()
{
	FILE* file = fopen( "assets/unity.tri", "r" );
	float a, b, c, d, e, f, g, h, i;
	for (int t = 0; t < N; t++)
	{
		fscanf( file, "%f %f %f %f %f %f %f %f %f\n",
			&a, &b, &c, &d, &e, &f, &g, &h, &i );
		tri[t].vertex0 = float3( a, b, c );
		tri[t].vertex1 = float3( d, e, f );
		tri[t].vertex2 = float3( g, h, i );
	}
	fclose( file );
	// construct the BVH
	// BuildBVH();
	BuildGrid();
}

//void FasterRaysApp::Init()
//{
//	// --- SCENE SELECTION ---
//	int scene_id = 3;
//
//	printf("Loading Scene %d...\n", scene_id);
//
//	// Reset the triangle counter
//	int triIdxCount = 0;
//
//	switch (scene_id)
//	{
//	case 1: // SCENE 1: Random cloud of triangles
//	{
//		// Create 10,000 random triangles
//		int count = 3000;
//		for (int i = 0; i < count; i++)
//		{
//			// Random positions in a box from -5 to 5
//			float3 r0(RandomFloat() * 10 - 5, RandomFloat() * 10 - 5, RandomFloat() * 10 - 5);
//			float3 r1(RandomFloat() * 10 - 5, RandomFloat() * 10 - 5, RandomFloat() * 10 - 5);
//			float3 r2(RandomFloat() * 10 - 5, RandomFloat() * 10 - 5, RandomFloat() * 10 - 5);
//
//			tri[triIdxCount].vertex0 = r0;
//			tri[triIdxCount].vertex1 = r1;
//			tri[triIdxCount].vertex2 = r2;
//			triIdxCount++;
//		}
//		break;
//	}
//
//	case 2: // SCENE 2: Dense cloud of triangles 
//	{
//		// 10,000 triangles packed in a TINY box 
//		int count = 3000;
//		for (int i = 0; i < count; i++)
//		{
//			// Box from -1 to 1 (very dense)
//			float3 r0(RandomFloat() * 2 - 1, RandomFloat() * 2 - 1, RandomFloat() * 2 - 1);
//			float3 r1(RandomFloat() * 2 - 1, RandomFloat() * 2 - 1, RandomFloat() * 2 - 1);
//			float3 r2(RandomFloat() * 2 - 1, RandomFloat() * 2 - 1, RandomFloat() * 2 - 1);
//
//			// push slight for camera visibility
//			float3 offset(0, 0, 1.5f);
//			tri[triIdxCount].vertex0 = r0 + offset;
//			tri[triIdxCount].vertex1 = r1 + offset;
//			tri[triIdxCount].vertex2 = r2 + offset;
//			triIdxCount++;
//		}
//		break;
//	}
//
//	case 3: // SCENE 3: The "Floor" (Structured Grid)
//	{
//		// 1. INCREASE SIZE
//		int gridSize = 100;
//		float scale = 0.05f;
//
//		// 2. center the grid
//		float offset = (gridSize * scale) / 2.0f;
//
//		for (int x = 0; x < gridSize; x++)
//		{
//			for (int z = 0; z < gridSize; z++)
//			{
//				// Calculate coordinates centered around 0,0
//				float x0 = x * scale - offset;
//				float x1 = (x + 1) * scale - offset;
//				float z0 = z * scale - offset;
//				float z1 = (z + 1) * scale - offset;
//
//				float3 p0 = float3(x0, 0, z0);
//				float3 p1 = float3(x1, 0, z0);
//				float3 p2 = float3(x1, 0, z1);
//				float3 p3 = float3(x0, 0, z1);
//
//				// Triangle 1
//				tri[triIdxCount].vertex0 = p0;
//				tri[triIdxCount].vertex1 = p1;
//				tri[triIdxCount].vertex2 = p2;
//				triIdxCount++;
//
//				// Triangle 2
//				tri[triIdxCount].vertex0 = p0;
//				tri[triIdxCount].vertex1 = p2;
//				tri[triIdxCount].vertex2 = p3;
//				triIdxCount++;
//			}
//		}
//		break;
//	}
//	}
//
//	// UPDATE THE GLOBAL N
//	N = triIdxCount;
//
//	// construct the BVH
//	BuildBVH();
//}



void FasterRaysApp::Tick( float deltaTime )
{
	// draw the scene
	screen->Clear( 0 );

	// -----------------------------------------------------------
	// CAMERA SELECTOR - Default scene
	// -----------------------------------------------------------
	// define the corners of the screen in worldspace
	float3 p0(-2.5f, 0.8f, -0.5f), p1(-0.5f, 0.8f, -0.5f), p2(-2.5f, -1.2f, -0.5f);

	// 1. Setup Camera
	float3 camPos(-1.5f, -0.2f, -2.5f);

	// -----------------------------------------------------------
	// CAMERA SELECTOR - Compare 3 scenes
	// -----------------------------------------------------------

	// CAM 1: For Scene 1 (The Big Cloud) - far back
	//float3 camPos(0, 0, -18.0f); // (Front View)
	// float3 camPos(15.0f, 0, 0); // (Side View)

	// CAM 2: For Scene 2 (The Dense Cluster) - close!
	// float3 camPos( 0, 0, -3.5f ); // (Close View)
	// float3 camPos( 0, 0, -8.0f ); // (Far View)

	// CAM 3: For Scene 3 (The Grid/Wall) - Go High and look down
	// float3 camPos( 0.0, 5.0f, -5.5f ); // (Angled View)
	// float3 camPos( 0, 5.0f, -15.0f ); // (Low View)
	// -----------------------------------------------------------

	// 1.5 DEFINE SCREEN RELATIVE TO CAMERA
	// Scene 1 and 2
	// Screen is 2 units wide, 2 units tall, 2 units in front of camera
	// float3 p0 = camPos + float3(-1.0f, 1.0f, 2.0f); // Top-Left
	// float3 p1 = camPos + float3(1.0f, 1.0f, 2.0f); // Top-Right
	// float3 p2 = camPos + float3(-1.0f, -1.0f, 2.0f); // Bottom-Left

	// Scene 3
	// float3 p0 = camPos + float3(-1.0f, -1.0f, 2.0f); // Top-Left (Tilted down)
	// float3 p1 = camPos + float3(1.0f, -1.0f, 2.0f); // Top-Right
	// float3 p2 = camPos + float3(-1.0f, -3.0f, 2.0f); // Bottom-Left

	
	// 2. Setup Variables to track the Stats for this Frame
	long long total_steps = 0;
	long long total_tests = 0;
	long long ray_count = 0;

	unsigned int min_steps = 999999, max_steps = 0;
	unsigned int min_tests = 999999, max_tests = 0;

	Ray ray;
	Timer t;
	// 3. render tiles of pixels (loop through every pixel)
	for (int y = 0; y < SCRHEIGHT; y += 4) for (int x = 0; x < SCRWIDTH; x += 4)
	{
		// render a single tile
		for (int v = 0; v < 4; v++) for (int u = 0; u < 4; u++)
		{
			// Reset counters for this specific ray
			global_traversal_steps = 0;
			global_intersect_tests = 0;

			// calculate the position of a pixel on the screen in worldspace
			float3 pixelPos = p0 + (p1 - p0) * ((x + u) / (float)SCRWIDTH) + (p2 - p0) * ((y + v) / (float)SCRHEIGHT);
			// define the ray in worldspace
			ray.O = camPos;
			ray.D = normalize( pixelPos - ray.O ), ray.t = 1e30f;
			// calculare reciprocal ray directions to speedup AABB intersections
			ray.rD = float3( 1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z );
			//  IntersectBVH( ray );
			IntersectGrid( ray );

			// GATHER DATA
			// Update Totals (for Average)
			total_steps += global_traversal_steps;
			total_tests += global_intersect_tests;
			ray_count++;

			// Update Mins and Maxes
			if (global_traversal_steps < min_steps) min_steps = global_traversal_steps;
			if (global_traversal_steps > max_steps) max_steps = global_traversal_steps;

			if (global_intersect_tests < min_tests) min_tests = global_intersect_tests;
			if (global_intersect_tests > max_tests) max_tests = global_intersect_tests;

			uint c = 500 - (int)(ray.t * 42);
			if (ray.t < 1e30f) screen->Plot( x + u, y + v, c * 0x10101 );
		}
	}
	float elapsed = t.elapsed() * 1000;
	printf( "tracing time: %.2fms (%5.2fK rays/s)\n", elapsed, sqr( 630 ) / elapsed );

	// 4. Calculate Averages
	float avg_steps = (float)total_steps / ray_count;
	float avg_tests = (float)total_tests / ray_count;

	// 5. Print results
	printf("VIEWPOINT 1 DATA:\n");
	printf("  Steps -> Avg: %.2f, Min: %d, Max: %d\n", avg_steps, min_steps, max_steps);
	printf("  Tests -> Avg: %.2f, Min: %d, Max: %d\n", avg_tests, min_tests, max_tests);
	printf("------------------------------------------------\n");
}

// EOF