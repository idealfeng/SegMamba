/***********************
 * ERA5-Land Winter Export (181 days, drop Feb-29)
 * Output: 2 GeoTIFF per winter
 *   - T2m in Celsius (float)
 *   - Precip in mm/day (float)
 * Bands: day_000 ... day_180
 ************************/

// =====================
// 0) USER SETTINGS
// =====================
var ROI = ee.Geometry.Rectangle([78, 22, 103, 45]); // Tibetan example
var OUT_PREFIX = 'Tibetan';                         // name in filename
var OUT_FOLDER = 'snow_persistence_era5l';          // Drive folder

// 15 winters: 2010-2011 ... 2024-2025
var START_WINTER_YEAR = 2010;
var END_WINTER_YEAR = 2024; // inclusive

// ERA5-Land DAILY_AGGR
var ERA5_COL = 'ECMWF/ERA5_LAND/DAILY_AGGR';
var BAND_TEMP = 'temperature_2m';
var BAND_PRCP = 'total_precipitation_sum';

// Export on ERA5 native grid (0.1 degree)
var ERA5_CRS = 'EPSG:4326';
var ERA5_TR = [0.1, 0, -180.05,
    0, -0.1, 90.05];

// Fixed 181-band names
var DAY_NAMES = ee.List.sequence(0, 180).map(function (i) {
    return ee.String('day_').cat(ee.Number(i).format('%03d'));
});

// =====================
// 1) Helpers
// =====================

// Build winter date list from Nov 1 to Apr 30 inclusive, endExclusive = May 1.
// Then drop Feb-29 if present, but keep Apr-30 (so size becomes 181 always).
function winterDatesNoFeb29(winterYear) {
    winterYear = ee.Number(winterYear);
    var start = ee.Date.fromYMD(winterYear, 11, 1);
    var endExcl = ee.Date.fromYMD(winterYear.add(1), 5, 1); // exclusive

    // list all days in [start, endExcl)
    var nDays = endExcl.difference(start, 'day'); // 181 or 182 (leap winter)
    var datesAll = ee.List.sequence(0, nDays.subtract(1)).map(function (k) {
        return start.advance(ee.Number(k), 'day');
    });

    // drop Feb-29
    var dates = datesAll.map(function (d) {
        d = ee.Date(d);
        var mmdd = d.format('MM-dd');
        return ee.Algorithms.If(mmdd.equals('02-29'), null, d);
    }).removeAll([null]);

    return {
        start: start,
        endExcl: endExcl,
        nDaysAll: nDays,
        datesAll: datesAll,
        dates: dates
    };
}

// Get daily image for a date [d, d+1), return single-band image (float).
// We also return a "missing flag" to diagnose if any day is absent.
function getEra5DailyBand(d, bandName, postFn) {
    d = ee.Date(d);
    var img = ee.ImageCollection(ERA5_COL)
        .filterDate(d, d.advance(1, 'day'))
        .first();

    var isMissing = ee.Algorithms.IsEqual(img, null);

    // If missing, fill with -9999 (should not happen normally, but keeps band count stable)
    var out = ee.Image(ee.Algorithms.If(
        isMissing,
        ee.Image.constant(-9999).toFloat(),
        postFn(ee.Image(img).select(bandName)).toFloat()
    ));

    return out.set({
        'system:time_start': d.millis(),
        'is_missing': isMissing
    });
}

// Stack 181 daily bands for a given winter
function stackWinter181(dates, bandName, postFn) {
    var imgs = ee.List(dates).map(function (d) {
        return getEra5DailyBand(d, bandName, postFn);
    });
    var ic = ee.ImageCollection.fromImages(imgs);
    var stacked = ic.toBands().rename(DAY_NAMES);
    var missingCount = ee.Number(ic.aggregate_sum('is_missing')); // true counts as 1
    return { image: stacked, missingCount: missingCount };
}

// =====================
// 2) Export per winter
// =====================
function exportOneWinter(y) {
    var info = winterDatesNoFeb29(y);
    var start = ee.Date(info.start);
    var endExcl = ee.Date(info.endExcl);
    var datesAll = ee.List(info.datesAll);
    var dates = ee.List(info.dates);
    var nAll = ee.Number(info.nDaysAll);

    // checks
    var hasFeb29 = datesAll.map(function (d) { return ee.Date(d).format('MM-dd'); })
        .filter(ee.Filter.eq('item', '02-29')).size();

    print('====================================================');
    print('[WINTER]', y + '-' + (y + 1));
    print('[CHK] allDays count (181 non-leap / 182 leap):', nAll);
    print('[CHK] has Feb-29 in allDays (0 or 1):', hasFeb29);
    print('[CHK] dates after drop Feb-29 (expect 181):', dates.size());
    print('[CHK] first day:', ee.Date(dates.get(0)).format('YYYY-MM-dd'));
    print('[CHK] last day :', ee.Date(dates.get(180)).format('YYYY-MM-dd'));
    print('[CHK] window:', start.format('YYYY-MM-dd'), 'to', endExcl.format('YYYY-MM-dd'), '(end exclusive)');

    // Build 181-band stacks
    var tempPack = stackWinter181(dates, BAND_TEMP, function (x) { return x.subtract(273.15); }); // K->C
    var prcpPack = stackWinter181(dates, BAND_PRCP, function (x) { return x.multiply(1000); });  // m->mm

    var temp181 = ee.Image(tempPack.image).clip(ROI);
    var prcp181 = ee.Image(prcpPack.image).clip(ROI);

    print('[CHK] temp181 bands (expect 181):', temp181.bandNames().size(), 'missingDays:', tempPack.missingCount);
    print('[CHK] prcp181 bands (expect 181):', prcp181.bandNames().size(), 'missingDays:', prcpPack.missingCount);

    // If you want to hard-stop when not 181, just visually check the console before running Tasks.
    // (GEE doesn't have a clean "throw" to stop exports.)

    var y0 = String(y);
    var y1 = String(y + 1);

    Export.image.toDrive({
        image: temp181,
        description: OUT_PREFIX + '_ERA5L_T2mC_native_' + y0 + '_' + y1 + '_181d',
        folder: OUT_FOLDER,
        region: ROI,
        crs: ERA5_CRS,
        crsTransform: ERA5_TR,
        maxPixels: 1e13
    });

    Export.image.toDrive({
        image: prcp181,
        description: OUT_PREFIX + '_ERA5L_PrcpMM_native_' + y0 + '_' + y1 + '_181d',
        folder: OUT_FOLDER,
        region: ROI,
        crs: ERA5_CRS,
        crsTransform: ERA5_TR,
        maxPixels: 1e13
    });
}

// =====================
// 3) Batch create tasks
// =====================
for (var y = START_WINTER_YEAR; y <= END_WINTER_YEAR; y++) {
    exportOneWinter(y);
}
