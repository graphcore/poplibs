#ifndef _vtilemapper_hpp_
#define _vtilemapper_hpp_

/* This class provides a tile mapper fo the IPU model engine based
   on "virtual tiles".
   Vertices are places on a virtual tiles which are then split evenly
   to cover the actual physical number of tiles. The split guarantees
   to split a virtual tile into contiguous blocks.

   Any vertices not placed on virtual tiles are then split evenly over the
   tiles as well.
*/
class VTileMapper {
  std::vector<std::vector<unsigned>> vTiles;
public:
  unsigned createVTile() {
    vTiles.push_back(std::vector<unsigned>());
    return vTiles.size() - 1;
  }

  void addToVTile(unsigned vTile, unsigned vertex_id) {
    vTiles[vTile].push_back(vertex_id);
  }

  std::vector<unsigned> createTileMapping(unsigned numVertices,
                                          unsigned numTiles) {
    unsigned invalidTile = numTiles + 1;
    std::vector<unsigned> map(numVertices, invalidTile);

    std::vector<unsigned> vTileOrdered;
    for (auto v : vTiles)
      vTileOrdered.insert(vTileOrdered.end(), v.begin(), v.end());

    unsigned numVTileVertices = vTileOrdered.size();
    unsigned numOtherVertices = numVertices - vTileOrdered.size();
    double vertsPerTile = (double) numVTileVertices / numTiles;
    double vertsPerTileFrac = vertsPerTile - (double ((unsigned) vertsPerTile));
    double remainder = 0;

    auto vTileIter = vTileOrdered.begin();
    for (unsigned i = 0; i < numTiles; i++) {
      unsigned vertsOnThisTile = (unsigned) vertsPerTile;
      remainder += vertsPerTileFrac;
      if (remainder > 1) {
        remainder -= 1;
        vertsOnThisTile += 1;
      }
      for (unsigned j = 0; j < vertsOnThisTile; ++j) {
        if (vTileIter != vTileOrdered.end()) {
          map[*vTileIter] = i;
          ++vTileIter;
        }
      }
    }

    vertsPerTile = (double) numOtherVertices / numTiles;
    vertsPerTileFrac = vertsPerTile - (double ((unsigned) vertsPerTile));
    remainder = 0;

    unsigned curVert = 0;
    for (unsigned i = 0; i < numTiles; i++) {
      unsigned vertsOnThisTile = (unsigned) vertsPerTile;
      remainder += vertsPerTileFrac;
      if (remainder > 1) {
        remainder -= 1;
        vertsOnThisTile += 1;
      }
      for (unsigned j = 0; j < vertsOnThisTile; ++j) {
        while (map[curVert] != invalidTile  && curVert < numVertices)
          curVert++;
        if (curVert < numVertices) {
          map[curVert] = i;
          curVert++;
        }
      }
    }

    for (unsigned j = curVert; j < numVertices; ++j) {
      if (map[j] == invalidTile)
        map[j] = 0;
    }

    return map;
  }
};

#endif //_vtilemapper_hpp_
