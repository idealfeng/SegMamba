// ============================================================================
// Tibetan 6ch OFFICIAL (NDSI+MASK per day) + 4 static
// 181 days fixed (Nov01-Apr30) dropping Feb29
// Export: 366 bands = 181*2 + 4
// Output layout (for local validate PASS): grouped NDSI then grouped MASK, then 4 static
// CRITICAL: make EVERY band Int16 + make EVERY pixel unmasked (avoid GeoTIFF NoData->0)
// ROI MUST NOT CHANGE
// ============================================================================

// -------------------- Config --------------------
var REGION_NAME = 'Tibetan';
var region = ee.Geometry.Rectangle([85, 30, 95, 36]);  // !!! ROI 不改
var START_YEAR = 2010;
var END_YEAR = 2024;   // 先只跑 2010，本地 validate 通过后再改大

var SCALE = 500;

// Fill values
var NDSI_FILL = 255;     // Int16 (MOD10A1 special)
var STATIC_FILL = -9999;   // Int16
var NDVI_FILL = -9999;   // Int16

var DO_QC = true;

// Data sources
var mod10 = ee.ImageCollection('MODIS/061/MOD10A1');
var mod13 = ee.ImageCollection('MODIS/061/MOD13A1');
var demImg = ee.Image('NASA/NASADEM_HGT/001');

// -------------------- Date list (181) --------------------
function buildDatesFixed181(startYear) {
    var start = ee.Date.fromYMD(startYear, 11, 1);
    var end = ee.Date.fromYMD(startYear + 1, 5, 1); // exclusive May 1
    var nDays = end.difference(start, 'day');         // 181 or 182 (leap season contains Feb29)

    var offsets = ee.List.sequence(0, nDays.subtract(1));
    var dates = offsets.map(function (k) {
        var d = start.advance(ee.Number(k), 'day');
        var m = ee.Number(d.get('month'));
        var dd = ee.Number(d.get('day'));
        var isFeb29 = m.eq(2).and(dd.eq(29));
        return ee.Algorithms.If(isFeb29, null, d.format('YYYYMMdd'));
    }).removeAll([null]);

    return ee.List(dates); // should be 181
}

// -------------------- Projection template --------------------
function getTemplateProjection(firstDateStr) {
    var dt = ee.Date.parse('YYYYMMdd', firstDateStr);
    var dayCol = mod10.filterBounds(region).filterDate(dt, dt.advance(1, 'day'));
    var img = ee.Image(ee.Algorithms.If(
        dayCol.size().gt(0),
        dayCol.first(),
        mod10.filterBounds(region).first()
    ));
    return img.select('NDSI_Snow_Cover').projection();
}

// -------------------- Utility: fully-filled base image (no mask) --------------------
function filledConstant(value, proj, bandName) {
    // A constant image that exists everywhere (no mask) in target projection.
    return ee.Image.constant(value)
        .toInt16()
        .rename(bandName)
        .reproject(proj);
}

// -------------------- Static bands (robust: NO masked pixels -> no GeoTIFF 0) --------------------
// Core rule:
//   - continuous fields: resample+reproject in float, then round, then Int16
//   - NEVER rely on unmask alone; build from a global filled base and "where(valid, value)"
function buildStaticBands(proj, startYear) {
    var dem = demImg.select('elevation').toFloat();

    // Elevation (meters)
    var elevVal = dem
        .resample('bilinear')
        .reproject(proj)
        .round()
        .toInt16();

    var elev = filledConstant(STATIC_FILL, proj, 'Elevation')
        .where(elevVal.mask(), elevVal)
        .toInt16();

    // Terrain products (slope/aspect) from DEM
    var terrain = ee.Terrain.products(dem);
    var slopeDeg = terrain.select('slope').toFloat();   // degrees
    var aspectDeg = terrain.select('aspect').toFloat();  // degrees

    // Slope_x100 (IMPORTANT: multiply AFTER reproject, to avoid <1 -> int -> 0)
    var slopeReproj = slopeDeg.resample('bilinear').reproject(proj);
    var slope_x100_val = slopeReproj.multiply(100).round().toInt16();

    var slope_x100 = filledConstant(STATIC_FILL, proj, 'Slope_x100')
        .where(slope_x100_val.mask(), slope_x100_val)
        .toInt16();

    // Northness_x10000 = cos(aspectRad) * 10000 in [-10000,10000]
    var northFloat = aspectDeg
        .multiply(Math.PI / 180)
        .cos()
        .resample('bilinear')
        .reproject(proj);

    var north_x10000_val = northFloat.multiply(10000).round().toInt16();

    var north_x10000 = filledConstant(STATIC_FILL, proj, 'Northness_x10000')
        .where(north_x10000_val.mask(), north_x10000_val)
        .toInt16();

    // NDVI max in season window (MOD13A1 is 16-day composite; we take max over window)
    var ndviStart = ee.Date.fromYMD(startYear, 11, 1);
    var ndviEnd = ee.Date.fromYMD(startYear + 1, 5, 1);

    var ndviCol = mod13.filterBounds(region).filterDate(ndviStart, ndviEnd).select('NDVI');

    var emptyMasked = ee.Image(0).updateMask(ee.Image(0)); // fully masked
    var ndviRaw = ee.Image(ee.Algorithms.If(
        ndviCol.size().gt(0),
        ndviCol.max(),
        emptyMasked
    )).toFloat();

    var ndviVal = ndviRaw.resample('bilinear').reproject(proj).round().toInt16();

    var ndvi = filledConstant(NDVI_FILL, proj, 'NDVI')
        .where(ndviVal.mask(), ndviVal)
        .toInt16();

    return ee.Image.cat([elev, slope_x100, north_x10000, ndvi])
        .toInt16()
        .setDefaultProjection(proj);
}

// -------------------- Daily dynamic (robust) --------------------
// Output per-day: 2 bands named exactly: 'NDSI' and 'MASK'
// Image ID (system:index) is date string 'YYYYMMdd' -> toBands names become 'YYYYMMdd_NDSI' / 'YYYYMMdd_MASK'
function buildDailyImage(ds, proj) {
    ds = ee.String(ds);
    var dt = ee.Date.parse('YYYYMMdd', ds);

    var dayCol = mod10.filterBounds(region).filterDate(dt, dt.advance(1, 'day'));
    var hasDay = dayCol.size().gt(0);

    var emptyMasked = ee.Image(0).updateMask(ee.Image(0)); // fully masked image
    var rawObs = ee.Image(ee.Algorithms.If(
        hasDay,
        ee.Image(dayCol.first()).select('NDSI_Snow_Cover'),
        emptyMasked
    )).toInt16();

    // Normalize specials: any code >=200 -> 255
    var obsClean = rawObs.where(rawObs.gte(200), NDSI_FILL).toInt16();

    // Build NDSI from global filled base; insert observation where it exists
    var ndsi = filledConstant(NDSI_FILL, proj, 'NDSI')
        .where(obsClean.mask(), obsClean)
        .toInt16();

    // MASK derived strictly from NDSI==255 (mask=1 valid, 0 invalid)
    var mask = ndsi.neq(NDSI_FILL).toInt16().rename('MASK');

    // Final per-day image (2 bands), stable projection, stable ID
    return ndsi.addBands(mask)
        .toInt16()
        .setDefaultProjection(proj)
        .set('system:index', ds)            // critical for toBands naming
        .set('system:time_start', dt.millis());
}

// -------------------- QC helpers (low-concurrency) --------------------
function meanNumber(img1band, bandName) {
    var d = img1band.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: region,
        scale: SCALE,
        maxPixels: 1e11
    });
    return ee.Number(d.get(bandName));
}

function qcStatic(static4, y) {
    // 8 indicators -> ONE reduceRegion (avoid "too many concurrent aggregations")
    var elev = static4.select('Elevation');
    var slope = static4.select('Slope_x100');
    var north = static4.select('Northness_x10000');
    var ndvi = static4.select('NDVI');

    var ind = ee.Image.cat([
        elev.eq(STATIC_FILL).rename('elevFill'),
        elev.eq(0).rename('elevZero'),
        slope.eq(STATIC_FILL).rename('slopeFill'),
        slope.eq(0).rename('slopeZero'),
        north.eq(STATIC_FILL).rename('northFill'),
        north.eq(0).rename('northZero'),
        ndvi.eq(NDVI_FILL).rename('ndviFill'),
        ndvi.eq(0).rename('ndviZero')
    ]);

    var meanDict = ind.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: region,
        scale: SCALE,
        maxPixels: 1e11
    });

    print('QC static mean (holes/fills) y=' + y, meanDict);

    // min/max for all 4 statics -> ONE reduceRegion
    var mm = static4.reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: region,
        scale: SCALE,
        maxPixels: 1e11
    });
    print('QC static min/max y=' + y, mm);
}

function qcDynamicFull(ndsiStack, maskStack, y) {
    // ndsiStack: 181 bands named YYYYMMdd_NDSI
    // maskStack: 181 bands named YYYYMMdd_MASK
    // 这里不使用 toArray / arrayReduce，避免 array 类型导致 reduceRegion 报错

    // per-pixel average over 181 days  -> scalar band
    var ndsiFill255_img = ndsiStack.eq(NDSI_FILL).reduce(ee.Reducer.mean()).rename('ndsiFill255');
    var ndsiGe200_img = ndsiStack.gte(200).reduce(ee.Reducer.mean()).rename('ndsiGe200'); // 理论上应≈ndsiFill255
    var ndsiIn0100_img = ndsiStack.gte(0).and(ndsiStack.lte(100)).reduce(ee.Reducer.mean()).rename('ndsiIn0_100');

    var mask0_img = maskStack.eq(0).reduce(ee.Reducer.mean()).rename('mask0');
    var mask1_img = maskStack.eq(1).reduce(ee.Reducer.mean()).rename('mask1');

    // mismatch: (mask==0) XOR (ndsi==255) over all days
    var mismatch_img = maskStack.eq(0).neq(ndsiStack.eq(NDSI_FILL))
        .reduce(ee.Reducer.mean()).rename('mismatch');

    // pseudo0: ndsi==0 AND mask==0 (should be 0 if落盘没把 masked 写成 0)
    var pseudo0_img = ndsiStack.eq(0).and(maskStack.eq(0))
        .reduce(ee.Reducer.mean()).rename('pseudo0');

    // 只做一次 reduceRegion（最稳，避免并发聚合过多）
    var qcImg = ee.Image.cat([
        mask0_img, mask1_img,
        mismatch_img,
        ndsiFill255_img, ndsiGe200_img, ndsiIn0100_img,
        pseudo0_img
    ]).toFloat();

    var dict = qcImg.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: region,
        scale: SCALE,
        maxPixels: 1e11
    });

    print('QC FULL y=' + y + ' (ratios over all pixels×days)', dict);

    // validDays stats（保持你原来的即可）
    var validDays = maskStack.reduce(ee.Reducer.sum()).rename('validDays'); // 0..181
    var stats = validDays.reduceRegion({
        reducer: ee.Reducer.mean()
            .combine({ reducer2: ee.Reducer.percentile([5, 25, 50, 75, 95]), sharedInputs: true }),
        geometry: region,
        scale: SCALE,
        maxPixels: 1e11
    });
    print('QC validDays stats y=' + y, stats);
}


function qcSampleDays(ndsiStack, maskStack, dateList, y) {
    // only 3 days: first/mid/last
    var d0 = ee.String(dateList.get(0));
    var d1 = ee.String(dateList.get(90));
    var d2 = ee.String(dateList.get(180));
    var picks = ee.List([d0, d1, d2]);

    var fc = ee.FeatureCollection(picks.map(function (ds) {
        ds = ee.String(ds);
        var bnN = ds.cat('_NDSI');
        var bnM = ds.cat('_MASK');

        var ndsi = ndsiStack.select([bnN]).rename('ndsi');
        var mask = maskStack.select([bnM]).rename('mask');

        var ndsiFill = meanNumber(ndsi.eq(NDSI_FILL).rename('m'), 'm');
        var mask0 = meanNumber(mask.eq(0).rename('m'), 'm');

        // mismatch per day: (mask==0) XOR (ndsi==255)
        var mismatch = meanNumber(mask.eq(0).neq(ndsi.eq(NDSI_FILL)).rename('m'), 'm');

        // pseudo0 per day: ndsi==0 AND mask==0
        var pseudo0 = meanNumber(ndsi.eq(0).and(mask.eq(0)).rename('m'), 'm');

        return ee.Feature(null, {
            day: ds,
            mask0: mask0,
            ndsiFill255: ndsiFill,
            mismatch: mismatch,
            pseudo0: pseudo0
        });
    }));

    print('QC sample days y=' + y, fc);
}

// -------------------- Build season --------------------
function buildSeasonFixed181(y) {
    var dateList = buildDatesFixed181(y);
    var dateCount = dateList.size();
    var feb29Count = ee.Number(dateList.filter(ee.Filter.stringContains('item', '0229')).size());

    print('==================== YEAR ' + y + ' ====================');
    print('QC dateCount (must be 181):', dateCount);
    print('QC hasFeb29 (must be 0):', feb29Count);
    print('QC season start/end:', dateList.get(0), dateList.get(180));

    var proj = getTemplateProjection(ee.String(dateList.get(0)));

    // static 4
    var static4 = buildStaticBands(proj, y);
    if (DO_QC) qcStatic(static4, y);

    // daily 181 images, each has bands ['NDSI','MASK'], image id = 'YYYYMMdd'
    var dailyIC = ee.ImageCollection.fromImages(
        dateList.map(function (ds) {
            return buildDailyImage(ds, proj);
        })
    ).sort('system:index');

    // toBands -> band names become 'YYYYMMdd_NDSI' and 'YYYYMMdd_MASK'
    var stacked = dailyIC.toBands().toInt16();

    // Explicit band lists (NO regex; no toBands-prefix trap)
    var ndsiNames = dateList.map(function (ds) { return ee.String(ds).cat('_NDSI'); });
    var maskNames = dateList.map(function (ds) { return ee.String(ds).cat('_MASK'); });

    var ndsiStack = stacked.select(ndsiNames).toInt16();
    var maskStack = stacked.select(maskNames).toInt16();

    // Grouped layout for local validate:
    // [181 NDSI] + [181 MASK]
    var out = ndsiStack.addBands(maskStack).toInt16();

    if (DO_QC) {
        qcSampleDays(ndsiStack, maskStack, dateList, y);
        qcDynamicFull(ndsiStack, maskStack, y);
    }

    // Final season image (do NOT clip / updateMask)
    var seasonImg = out.addBands(static4).toInt16().setDefaultProjection(proj);

    // Strong sanity checks
    print('bandCount (must be 366):', seasonImg.bandNames().size());
    print('bandTypes (all should be Int16):', seasonImg.bandTypes());

    return seasonImg;
}

// -------------------- Export loop --------------------
var folderName = REGION_NAME + '_SnowCover_6ch_fixed181_OFFICIAL';

for (var y = START_YEAR; y <= END_YEAR; y++) {
    var desc = REGION_NAME + '_SnowCover_' + y + '_' + (y + 1) + '_6ch_fixed181_OFFICIAL';

    var seasonImg = buildSeasonFixed181(y);

    Export.image.toDrive({
        image: seasonImg,
        description: desc,
        folder: folderName,
        scale: SCALE,
        region: region,
        fileFormat: 'GeoTIFF',
        maxPixels: 1e11,
        formatOptions: { cloudOptimized: true }
    });
}

// Map preview
Map.setCenter(90, 33, 6);
var outline = ee.Image().byte().paint({
    featureCollection: ee.FeatureCollection([ee.Feature(region)]),
    color: 1,
    width: 3
});
Map.addLayer(outline, { palette: ['red'] }, 'ROI outline');
