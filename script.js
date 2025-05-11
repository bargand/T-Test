document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements
  const testTypeSelect = document.getElementById("testType");
  const inputForm = document.getElementById("inputForm");
  const tTestForm = document.getElementById("tTestForm");
  const resultsDiv = document.getElementById("results");
  const resultContent = document.getElementById("resultContent");
  const calculateBtn = document.getElementById("calculateBtn");
  const newTestBtn = document.getElementById("newTestBtn");
  const nullHypothesis = document.getElementById("nullHypothesis");
  const altHypothesis = document.getElementById("altHypothesis");
  const tailType = document.getElementById("tailType");
  const significanceLevel = document.getElementById("significanceLevel");
  const confidenceLevel = document.getElementById("confidenceLevel");

  // Constants
  const TEST_TYPES = {
    ONE_SAMPLE: "oneSample",
    TWO_SAMPLE: "twoSample",
    PAIRED: "paired",
    CI_SINGLE_MEAN: "ciSingleMean",
    CI_TWO_MEANS: "ciTwoMeans",
    CI_PAIRED: "ciPaired",
  };

  const TAIL_TYPES = {
    TWO_TAILED: "twoTailed",
    LEFT_TAILED: "leftTailed",
    RIGHT_TAILED: "rightTailed",
  };

  // ===================================
  // ACCURATE STATISTICAL FUNCTIONS
  // ===================================

  // Student's t-distribution CDF using accurate approximation
  function tCDF(t, df) {
    if (df <= 0) return NaN;

    const x = df / (t * t + df);
    const a = df / 2;
    const b = 0.5;

    // Regularized incomplete Beta function
    const ibeta = (x, a, b) => {
      const epsilon = 1e-10;
      const maxIter = 1000;

      if (x < 0 || x > 1 || a <= 0 || b <= 0) return NaN;
      if (x === 0) return 0;
      if (x === 1) return 1;

      // Continued fraction expansion
      let aplusb = a + b;
      let aplus1 = a + 1;
      let aminus1 = a - 1;
      let c = 1;
      let d = 1 - (aplusb * x) / aplus1;
      if (Math.abs(d) < epsilon) d = epsilon;
      d = 1 / d;
      let h = d;

      for (let m = 1; m <= maxIter; m++) {
        const m2 = 2 * m;
        let aa = (m * (b - m) * x) / ((aminus1 + m2) * (a + m2));
        d = 1 + aa * d;
        if (Math.abs(d) < epsilon) d = epsilon;
        c = 1 + aa / c;
        if (Math.abs(c) < epsilon) c = epsilon;
        d = 1 / d;
        h *= d * c;

        aa = (-(a + m) * (aplusb + m) * x) / ((a + m2) * (aplus1 + m2));
        d = 1 + aa * d;
        if (Math.abs(d) < epsilon) d = epsilon;
        c = 1 + aa / c;
        if (Math.abs(c) < epsilon) c = epsilon;
        d = 1 / d;
        const del = d * c;
        h *= del;

        if (Math.abs(del - 1) < epsilon) break;
      }

      // Using accurate logBeta implementation
      const logBeta = (a, b) => {
        // Accurate logGamma implementation
        const logGamma = (z) => {
          if (z < 0) return NaN;

          // Lanczos approximation coefficients
          const g = 7;
          const p = [
            0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
          ];

          if (z < 0.5) {
            return (
              Math.log(Math.PI) -
              Math.log(Math.sin(Math.PI * z)) -
              logGamma(1 - z)
            );
          }

          z -= 1;
          let x = p[0];
          for (let i = 1; i < p.length; i++) {
            x += p[i] / (z + i);
          }
          const t = z + g + 0.5;
          return (
            0.5 * Math.log(2 * Math.PI) +
            (z + 0.5) * Math.log(t) -
            t +
            Math.log(x)
          );
        };

        return logGamma(a) + logGamma(b) - logGamma(a + b);
      };

      return Math.exp(
        Math.log(h) +
          a * Math.log(x) +
          b * Math.log(1 - x) -
          Math.log(a) -
          logBeta(a, b)
      );
    };

    const ibetaVal = ibeta(x, a, b);
    return t > 0 ? 1 - ibetaVal / 2 : ibetaVal / 2;
  }

  // Inverse t-distribution using accurate Newton-Raphson method
  function inverseTCDF(p, df) {
    if (df <= 0 || p <= 0 || p >= 1) return NaN;

    // Initial approximation using inverse normal
    let t = inverseNormalCDF(p);
    if (isNaN(t)) t = 0;

    const maxIter = 100;
    const tolerance = 1e-10;

    for (let i = 0; i < maxIter; i++) {
      const cdf = tCDF(t, df);
      const diff = cdf - p;
      if (Math.abs(diff) < tolerance) break;

      // PDF calculation
      const pdf = Math.exp(
        logGamma((df + 1) / 2) -
          logGamma(df / 2) -
          0.5 * Math.log(Math.PI * df) -
          ((df + 1) / 2) * Math.log(1 + (t * t) / df)
      );

      t -= diff / pdf;
    }

    return t;
  }

  // Inverse normal CDF (Probit function)
  function inverseNormalCDF(p) {
    // Coefficients in rational approximations
    const a = [
      -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
      1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
    ];

    const b = [
      -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
      6.680131188771972e1, -1.328068155288572e1,
    ];

    const c = [
      -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
      -2.549732539343734, 4.374664141464968, 2.938163982698783,
    ];

    const d = [
      7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
      3.754408661907416,
    ];

    // Define break-points
    const plow = 0.02425;
    const phigh = 1 - plow;

    if (p < plow) {
      // Rational approximation for lower region
      const q = Math.sqrt(-2 * Math.log(p));
      return (
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
        c[5] / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
      );
    } else if (p <= phigh) {
      // Rational approximation for central region
      const q = p - 0.5;
      const r = q * q;
      return (
        ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r +
        (a[5] * q) /
          (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
      );
    } else {
      // Rational approximation for upper region
      const q = Math.sqrt(-2 * Math.log(1 - p));
      return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
        c[5] / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
      );
    }
  }

  // Log Gamma function (used in other calculations)
  function logGamma(z) {
    if (z < 0) return NaN;

    // Lanczos approximation coefficients
    const g = 7;
    const p = [
      0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];

    if (z < 0.5) {
      return (
        Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z)
      );
    }

    z -= 1;
    let x = p[0];
    for (let i = 1; i < p.length; i++) {
      x += p[i] / (z + i);
    }
    const t = z + g + 0.5;
    return (
      0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x)
    );
  }

  // Calculate p-value based on t-score and tail type
  function calculatePValue(tScore, df, tail) {
    const absT = Math.abs(tScore);
    let pValue = 2 * (1 - tCDF(absT, df));

    switch (tail) {
      case TAIL_TYPES.LEFT_TAILED:
        pValue = tScore < 0 ? pValue / 2 : 1 - pValue / 2;
        break;
      case TAIL_TYPES.RIGHT_TAILED:
        pValue = tScore > 0 ? pValue / 2 : 1 - pValue / 2;
        break;
    }

    return pValue;
  }

  // Get critical value based on alpha, degrees of freedom, and tail type
  function getCriticalValue(alpha, df, tail) {
    switch (tail) {
      case TAIL_TYPES.LEFT_TAILED:
        return inverseTCDF(alpha, df);
      case TAIL_TYPES.RIGHT_TAILED:
        return inverseTCDF(1 - alpha, df);
      case TAIL_TYPES.TWO_TAILED:
        return Math.abs(inverseTCDF(alpha / 2, df));
      default:
        throw new Error("Invalid tail type specified");
    }
  }

  // ===================================
  // APPLICATION CORE FUNCTIONS
  // ===================================

  // Initialize the application
  function init() {
    setupEventListeners();
    populateConfidenceLevels();
  }

  // Populate confidence level options
  function populateConfidenceLevels() {
    const levels = [80, 85, 90, 95, 99];
    confidenceLevel.innerHTML = levels
      .map((level) => `<option value="${level}">${level}%</option>`)
      .join("");
  }

  // Set up all event listeners
  function setupEventListeners() {
    testTypeSelect.addEventListener("change", handleTestTypeChange);
    tailType.addEventListener("change", updateHypothesisText);
    calculateBtn.addEventListener("click", handleCalculateClick);
    newTestBtn.addEventListener("click", resetCalculator);
  }

  // Handle test type selection change
  function handleTestTypeChange() {
    const selectedTest = testTypeSelect.value;

    if (selectedTest) {
      inputForm.classList.remove("hidden");
      generateInputFields(selectedTest);
      updateHypothesisText();

      // Show/hide appropriate sections
      const isConfidenceInterval = selectedTest.startsWith("ci");
      document
        .getElementById("hypothesisSection")
        .classList.toggle("hidden", isConfidenceInterval);
      document
        .getElementById("confidenceSection")
        .classList.toggle("hidden", !isConfidenceInterval);
    } else {
      inputForm.classList.add("hidden");
    }

    resultsDiv.classList.add("hidden");
  }

  // Generate input fields based on selected test type
  function generateInputFields(testType) {
    tTestForm.innerHTML = "";

    switch (testType) {
      case TEST_TYPES.ONE_SAMPLE:
        tTestForm.innerHTML = `
                    <div class="form-group">
                        <label for="sampleMean">Sample Mean (x̄)</label>
                        <input type="number" id="sampleMean" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="popMean">Hypothesized Mean (μ₀)</label>
                        <input type="number" id="popMean" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="sampleStdDev">Sample Standard Deviation (s)</label>
                        <input type="number" id="sampleStdDev" min="0.0001" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="sampleSize">Sample Size (n)</label>
                        <input type="number" id="sampleSize" min="2" required>
                    </div>
                `;
        break;

      case TEST_TYPES.TWO_SAMPLE:
        tTestForm.innerHTML = `
                    <div class="form-group">
                        <label for="sampleMean1">Sample 1 Mean (x̄₁)</label>
                        <input type="number" id="sampleMean1" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="sampleMean2">Sample 2 Mean (x̄₂)</label>
                        <input type="number" id="sampleMean2" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="sampleStdDev1">Sample 1 Standard Deviation (s₁)</label>
                        <input type="number" id="sampleStdDev1" min="0.0001" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="sampleStdDev2">Sample 2 Standard Deviation (s₂)</label>
                        <input type="number" id="sampleStdDev2" min="0.0001" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="sampleSize1">Sample 1 Size (n₁)</label>
                        <input type="number" id="sampleSize1" min="2" required>
                    </div>
                    <div class="form-group">
                        <label for="sampleSize2">Sample 2 Size (n₂)</label>
                        <input type="number" id="sampleSize2" min="2" required>
                    </div>
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="equalVariance"> Assume equal variances
                        </label>
                    </div>
                `;
        break;

      case TEST_TYPES.PAIRED:
        tTestForm.innerHTML = `
                    <div class="data-input">
                        <label>Before Treatment Data (comma separated)</label>
                        <input type="text" id="beforeData" placeholder="e.g., 10, 12, 14, 15">
                    </div>
                    <div class="data-input">
                        <label>After Treatment Data (comma separated)</label>
                        <input type="text" id="afterData" placeholder="e.g., 8, 9, 11, 12">
                    </div>
                `;
        break;

      case TEST_TYPES.CI_SINGLE_MEAN:
        tTestForm.innerHTML = `
                    <div class="form-group">
                        <label for="ciSampleMean">Sample Mean (x̄)</label>
                        <input type="number" id="ciSampleMean" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="ciSampleStdDev">Sample Standard Deviation (s)</label>
                        <input type="number" id="ciSampleStdDev" min="0.0001" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="ciSampleSize">Sample Size (n)</label>
                        <input type="number" id="ciSampleSize" min="2" required>
                    </div>
                `;
        break;

      case TEST_TYPES.CI_TWO_MEANS:
        tTestForm.innerHTML = `
                    <div class="form-group">
                        <label for="ciSampleMean1">Sample 1 Mean (x̄₁)</label>
                        <input type="number" id="ciSampleMean1" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="ciSampleMean2">Sample 2 Mean (x̄₂)</label>
                        <input type="number" id="ciSampleMean2" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="ciSampleStdDev1">Sample 1 Standard Deviation (s₁)</label>
                        <input type="number" id="ciSampleStdDev1" min="0.0001" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="ciSampleStdDev2">Sample 2 Standard Deviation (s₂)</label>
                        <input type="number" id="ciSampleStdDev2" min="0.0001" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="ciSampleSize1">Sample 1 Size (n₁)</label>
                        <input type="number" id="ciSampleSize1" min="2" required>
                    </div>
                    <div class="form-group">
                        <label for="ciSampleSize2">Sample 2 Size (n₂)</label>
                        <input type="number" id="ciSampleSize2" min="2" required>
                    </div>
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="ciEqualVariance"> Assume equal variances
                        </label>
                    </div>
                `;
        break;

      case TEST_TYPES.CI_PAIRED:
        tTestForm.innerHTML = `
                    <div class="data-input">
                        <label>First Measurement Data (comma separated)</label>
                        <input type="text" id="firstData" placeholder="e.g., 10, 12, 14, 15">
                    </div>
                    <div class="data-input">
                        <label>Second Measurement Data (comma separated)</label>
                        <input type="text" id="secondData" placeholder="e.g., 8, 9, 11, 12">
                    </div>
                `;
        break;
    }
  }

  // Update hypothesis text based on selected test and tail type
  function updateHypothesisText() {
    const testType = testTypeSelect.value;
    const tail = tailType.value;

    if (!testType || testType.startsWith("ci")) return;

    let nullText = "";
    let altText = "";

    switch (testType) {
      case TEST_TYPES.ONE_SAMPLE:
        nullText = "μ = μ₀ (population mean equals hypothesized value)";
        switch (tail) {
          case TAIL_TYPES.LEFT_TAILED:
            altText =
              "μ < μ₀ (population mean is less than hypothesized value)";
            break;
          case TAIL_TYPES.RIGHT_TAILED:
            altText =
              "μ > μ₀ (population mean is greater than hypothesized value)";
            break;
          default:
            altText =
              "μ ≠ μ₀ (population mean differs from hypothesized value)";
        }
        break;

      case TEST_TYPES.TWO_SAMPLE:
        nullText = "μ₁ = μ₂ (population means are equal)";
        switch (tail) {
          case TAIL_TYPES.LEFT_TAILED:
            altText =
              "μ₁ < μ₂ (population 1 mean is less than population 2 mean)";
            break;
          case TAIL_TYPES.RIGHT_TAILED:
            altText =
              "μ₁ > μ₂ (population 1 mean is greater than population 2 mean)";
            break;
          default:
            altText = "μ₁ ≠ μ₂ (population means are not equal)";
        }
        break;

      case TEST_TYPES.PAIRED:
        nullText = "μ_d = 0 (mean difference is zero)";
        switch (tail) {
          case TAIL_TYPES.LEFT_TAILED:
            altText = "μ_d < 0 (mean difference is negative)";
            break;
          case TAIL_TYPES.RIGHT_TAILED:
            altText = "μ_d > 0 (mean difference is positive)";
            break;
          default:
            altText = "μ_d ≠ 0 (mean difference is not zero)";
        }
        break;
    }

    nullHypothesis.textContent = nullText;
    altHypothesis.textContent = altText;
  }

  // Handle calculate button click
  function handleCalculateClick() {
    try {
      const testType = testTypeSelect.value;

      if (!testType) {
        throw new Error("Please select a test type");
      }

      if (testType.startsWith("ci")) {
        calculateConfidenceInterval(testType);
      } else {
        calculateHypothesisTest(testType);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
      console.error(error);
    }
  }

  // Calculate hypothesis test
  function calculateHypothesisTest(testType) {
    const alpha = parseFloat(significanceLevel.value);
    const tail = tailType.value;

    // Validate alpha
    if (isNaN(alpha) || alpha <= 0 || alpha >= 1) {
      throw new Error("Significance level must be between 0 and 1");
    }

    let tScore, pValue, degreesOfFreedom;

    switch (testType) {
      case TEST_TYPES.ONE_SAMPLE:
        ({ tScore, degreesOfFreedom } = calculateOneSampleTTest());
        break;

      case TEST_TYPES.TWO_SAMPLE:
        ({ tScore, degreesOfFreedom } = calculateTwoSampleTTest());
        break;

      case TEST_TYPES.PAIRED:
        ({ tScore, degreesOfFreedom } = calculatePairedTTest());
        break;
    }

    // Calculate p-value based on tail type
    pValue = calculatePValue(tScore, degreesOfFreedom, tail);

    // Determine if result is significant
    const isSignificant = pValue < alpha;

    // Get critical value
    const criticalValue = getCriticalValue(alpha, degreesOfFreedom, tail);

    // Display results
    displayHypothesisResults(
      testType,
      tScore,
      pValue,
      alpha,
      isSignificant,
      criticalValue,
      tail,
      degreesOfFreedom
    );

    // Show results section
    resultsDiv.classList.remove("hidden");
  }

  // Calculate confidence interval
  function calculateConfidenceInterval(testType) {
    const confidence = parseInt(confidenceLevel.value);
    const alpha = (100 - confidence) / 100;

    let result;

    switch (testType) {
      case TEST_TYPES.CI_SINGLE_MEAN:
        result = calculateSingleMeanCI(alpha);
        break;

      case TEST_TYPES.CI_TWO_MEANS:
        result = calculateTwoMeansCI(alpha);
        break;

      case TEST_TYPES.CI_PAIRED:
        result = calculatePairedCI(alpha);
        break;
    }

    // Display results
    displayCIResults(testType, confidence, result);

    // Show results section
    resultsDiv.classList.remove("hidden");
  }

  // Calculate one-sample t-test
  function calculateOneSampleTTest() {
    const sampleMean = parseFloat(document.getElementById("sampleMean").value);
    const popMean = parseFloat(document.getElementById("popMean").value);
    const sampleStdDev = parseFloat(
      document.getElementById("sampleStdDev").value
    );
    const sampleSize = parseInt(document.getElementById("sampleSize").value);

    // Validate inputs
    if (isNaN(sampleMean)) throw new Error("Sample mean must be a number");
    if (isNaN(popMean)) throw new Error("Hypothesized mean must be a number");
    if (isNaN(sampleStdDev) || sampleStdDev <= 0)
      throw new Error("Sample standard deviation must be a positive number");
    if (isNaN(sampleSize)) throw new Error("Sample size must be a number");
    if (sampleSize < 2) throw new Error("Sample size must be at least 2");

    const standardError = sampleStdDev / Math.sqrt(sampleSize);
    const tScore = (sampleMean - popMean) / standardError;
    const degreesOfFreedom = sampleSize - 1;

    return { tScore, degreesOfFreedom };
  }

  // Calculate two-sample t-test
  function calculateTwoSampleTTest() {
    const sampleMean1 = parseFloat(
      document.getElementById("sampleMean1").value
    );
    const sampleMean2 = parseFloat(
      document.getElementById("sampleMean2").value
    );
    const sampleStdDev1 = parseFloat(
      document.getElementById("sampleStdDev1").value
    );
    const sampleStdDev2 = parseFloat(
      document.getElementById("sampleStdDev2").value
    );
    const sampleSize1 = parseInt(document.getElementById("sampleSize1").value);
    const sampleSize2 = parseInt(document.getElementById("sampleSize2").value);
    const equalVariance = document.getElementById("equalVariance").checked;

    // Validate inputs
    if (isNaN(sampleMean1)) throw new Error("Sample 1 mean must be a number");
    if (isNaN(sampleMean2)) throw new Error("Sample 2 mean must be a number");
    if (isNaN(sampleStdDev1) || sampleStdDev1 <= 0)
      throw new Error("Sample 1 standard deviation must be a positive number");
    if (isNaN(sampleStdDev2) || sampleStdDev2 <= 0)
      throw new Error("Sample 2 standard deviation must be a positive number");
    if (isNaN(sampleSize1)) throw new Error("Sample 1 size must be a number");
    if (isNaN(sampleSize2)) throw new Error("Sample 2 size must be a number");
    if (sampleSize1 < 2) throw new Error("Sample 1 size must be at least 2");
    if (sampleSize2 < 2) throw new Error("Sample 2 size must be at least 2");

    let tScore, degreesOfFreedom;

    if (equalVariance) {
      // Pooled variance t-test
      const pooledVariance =
        ((sampleSize1 - 1) * Math.pow(sampleStdDev1, 2) +
          (sampleSize2 - 1) * Math.pow(sampleStdDev2, 2)) /
        (sampleSize1 + sampleSize2 - 2);
      const standardError = Math.sqrt(
        pooledVariance * (1 / sampleSize1 + 1 / sampleSize2)
      );
      tScore = (sampleMean1 - sampleMean2) / standardError;
      degreesOfFreedom = sampleSize1 + sampleSize2 - 2;
    } else {
      // Welch's t-test
      const variance1 = Math.pow(sampleStdDev1, 2) / sampleSize1;
      const variance2 = Math.pow(sampleStdDev2, 2) / sampleSize2;
      const standardError = Math.sqrt(variance1 + variance2);
      tScore = (sampleMean1 - sampleMean2) / standardError;

      // Welch-Satterthwaite equation for degrees of freedom
      degreesOfFreedom =
        Math.pow(variance1 + variance2, 2) /
        (Math.pow(variance1, 2) / (sampleSize1 - 1) +
          Math.pow(variance2, 2) / (sampleSize2 - 1));
    }

    return { tScore, degreesOfFreedom };
  }

  // Calculate paired t-test
  function calculatePairedTTest() {
    const beforeData = parseDataInput("beforeData");
    const afterData = parseDataInput("afterData");

    if (beforeData.length !== afterData.length) {
      throw new Error(
        "Paired data sets must have the same number of observations"
      );
    }
    if (beforeData.length < 2) {
      throw new Error("At least 2 pairs are required for a paired t-test");
    }

    // Calculate differences
    const differences = beforeData.map((val, i) => val - afterData[i]);
    const meanDiff = calculateMean(differences);
    const stdDevDiff = calculateStdDev(differences);
    const standardError = stdDevDiff / Math.sqrt(differences.length);
    const tScore = meanDiff / standardError;
    const degreesOfFreedom = differences.length - 1;

    return { tScore, degreesOfFreedom };
  }

  // Calculate single mean confidence interval
  function calculateSingleMeanCI(alpha) {
    const sampleMean = parseFloat(
      document.getElementById("ciSampleMean").value
    );
    const sampleStdDev = parseFloat(
      document.getElementById("ciSampleStdDev").value
    );
    const sampleSize = parseInt(document.getElementById("ciSampleSize").value);

    // Validate inputs
    if (isNaN(sampleMean)) throw new Error("Sample mean must be a number");
    if (isNaN(sampleStdDev) || sampleStdDev <= 0)
      throw new Error("Sample standard deviation must be a positive number");
    if (isNaN(sampleSize)) throw new Error("Sample size must be a number");
    if (sampleSize < 2) throw new Error("Sample size must be at least 2");

    const degreesOfFreedom = sampleSize - 1;
    const tCritical = getCriticalValue(
      alpha / 2,
      degreesOfFreedom,
      TAIL_TYPES.TWO_TAILED
    );
    const marginOfError = tCritical * (sampleStdDev / Math.sqrt(sampleSize));

    return {
      lower: sampleMean - marginOfError,
      upper: sampleMean + marginOfError,
      marginOfError: marginOfError,
      tCritical: tCritical,
      degreesOfFreedom: degreesOfFreedom,
    };
  }

  // Calculate two means confidence interval
  function calculateTwoMeansCI(alpha) {
    const sampleMean1 = parseFloat(
      document.getElementById("ciSampleMean1").value
    );
    const sampleMean2 = parseFloat(
      document.getElementById("ciSampleMean2").value
    );
    const sampleStdDev1 = parseFloat(
      document.getElementById("ciSampleStdDev1").value
    );
    const sampleStdDev2 = parseFloat(
      document.getElementById("ciSampleStdDev2").value
    );
    const sampleSize1 = parseInt(
      document.getElementById("ciSampleSize1").value
    );
    const sampleSize2 = parseInt(
      document.getElementById("ciSampleSize2").value
    );
    const equalVariance = document.getElementById("ciEqualVariance").checked;

    // Validate inputs
    if (isNaN(sampleMean1)) throw new Error("Sample 1 mean must be a number");
    if (isNaN(sampleMean2)) throw new Error("Sample 2 mean must be a number");
    if (isNaN(sampleStdDev1) || sampleStdDev1 <= 0)
      throw new Error("Sample 1 standard deviation must be a positive number");
    if (isNaN(sampleStdDev2) || sampleStdDev2 <= 0)
      throw new Error("Sample 2 standard deviation must be a positive number");
    if (isNaN(sampleSize1)) throw new Error("Sample 1 size must be a number");
    if (isNaN(sampleSize2)) throw new Error("Sample 2 size must be a number");
    if (sampleSize1 < 2) throw new Error("Sample 1 size must be at least 2");
    if (sampleSize2 < 2) throw new Error("Sample 2 size must be at least 2");

    let tCritical, degreesOfFreedom, standardError;
    const difference = sampleMean1 - sampleMean2;

    if (equalVariance) {
      // Pooled variance
      const pooledVariance =
        ((sampleSize1 - 1) * Math.pow(sampleStdDev1, 2) +
          (sampleSize2 - 1) * Math.pow(sampleStdDev2, 2)) /
        (sampleSize1 + sampleSize2 - 2);
      standardError = Math.sqrt(
        pooledVariance * (1 / sampleSize1 + 1 / sampleSize2)
      );
      degreesOfFreedom = sampleSize1 + sampleSize2 - 2;
    } else {
      // Welch's t-test
      const variance1 = Math.pow(sampleStdDev1, 2) / sampleSize1;
      const variance2 = Math.pow(sampleStdDev2, 2) / sampleSize2;
      standardError = Math.sqrt(variance1 + variance2);

      // Welch-Satterthwaite equation for degrees of freedom
      degreesOfFreedom =
        Math.pow(variance1 + variance2, 2) /
        (Math.pow(variance1, 2) / (sampleSize1 - 1) +
          Math.pow(variance2, 2) / (sampleSize2 - 1));
    }

    tCritical = getCriticalValue(
      alpha / 2,
      degreesOfFreedom,
      TAIL_TYPES.TWO_TAILED
    );
    const marginOfError = tCritical * standardError;

    return {
      lower: difference - marginOfError,
      upper: difference + marginOfError,
      marginOfError: marginOfError,
      tCritical: tCritical,
      degreesOfFreedom: degreesOfFreedom,
    };
  }

  // Calculate paired confidence interval
  function calculatePairedCI(alpha) {
    const firstData = parseDataInput("firstData");
    const secondData = parseDataInput("secondData");

    if (firstData.length !== secondData.length) {
      throw new Error(
        "Paired data sets must have the same number of observations"
      );
    }
    if (firstData.length < 2) {
      throw new Error("At least 2 pairs are required");
    }

    // Calculate differences
    const differences = firstData.map((val, i) => val - secondData[i]);
    const meanDiff = calculateMean(differences);
    const stdDevDiff = calculateStdDev(differences);
    const degreesOfFreedom = differences.length - 1;
    const tCritical = getCriticalValue(
      alpha / 2,
      degreesOfFreedom,
      TAIL_TYPES.TWO_TAILED
    );
    const marginOfError =
      tCritical * (stdDevDiff / Math.sqrt(differences.length));

    return {
      lower: meanDiff - marginOfError,
      upper: meanDiff + marginOfError,
      marginOfError: marginOfError,
      tCritical: tCritical,
      degreesOfFreedom: degreesOfFreedom,
    };
  }

  // Helper function to parse comma-separated data input
  function parseDataInput(elementId) {
    const input = document.getElementById(elementId).value;
    return input
      .split(",")
      .map((item) => parseFloat(item.trim()))
      .filter((item) => !isNaN(item));
  }

  // Helper function to calculate mean
  function calculateMean(data) {
    if (data.length === 0) return NaN;
    return data.reduce((sum, val) => sum + val, 0) / data.length;
  }

  // Helper function to calculate standard deviation
  function calculateStdDev(data) {
    if (data.length < 2) return NaN;
    const mean = calculateMean(data);
    const squaredDiffs = data.map((val) => Math.pow(val - mean, 2));
    const variance =
      squaredDiffs.reduce((sum, val) => sum + val, 0) / (data.length - 1);
    return Math.sqrt(variance);
  }

  // Display hypothesis test results
  function displayHypothesisResults(
    testType,
    tScore,
    pValue,
    alpha,
    isSignificant,
    criticalValue,
    tail,
    df
  ) {
    const testName = getTestName(testType);
    const comparison = getComparisonText(criticalValue, tail);
    const tailDescription = getTailDescription(tail);

    resultContent.innerHTML = `
            <div class="result-item">
                <strong>Test Performed:</strong> ${testName}
            </div>
            <div class="result-item">
                <strong>Degrees of Freedom:</strong> ${df.toFixed(2)}
            </div>
            <div class="result-item">
                <strong>T-Score:</strong> ${tScore.toFixed(4)}
            </div>
            <div class="result-item">
                <strong>p-value:</strong> ${
                  pValue < 0.0001 ? pValue.toExponential(4) : pValue.toFixed(4)
                }
            </div>
            <div class="result-item">
                <strong>Significance Level (α):</strong> ${alpha}
            </div>
            <div class="result-item">
                <strong>Critical Value:</strong> ${comparison}
            </div>
            <div class="result-item ${
              isSignificant ? "significant" : "not-significant"
            }">
                <strong>Conclusion:</strong> ${
                  isSignificant
                    ? "Reject the null hypothesis (H₀). The result is statistically significant."
                    : "Fail to reject the null hypothesis (H₀). The result is not statistically significant."
                }
            </div>
            <div class="critical-value">
                <strong>Interpretation:</strong> 
                <p>A ${tailDescription} test was performed at α = ${alpha} level of significance with ${df.toFixed(
      2
    )} degrees of freedom.</p>
                <p>The calculated p-value (${
                  pValue < 0.0001 ? pValue.toExponential(4) : pValue.toFixed(4)
                }) is ${pValue < alpha ? "less" : "greater"} than α.</p>
                <p>T-score of ${tScore.toFixed(4)} ${
      isSignificant
        ? "falls in the critical region"
        : "does not fall in the critical region"
    }.</p>
            </div>
        `;
  }

  // Display confidence interval results
  function displayCIResults(testType, confidence, result) {
    const testName = getTestName(testType);

    resultContent.innerHTML = `
            <div class="result-item">
                <strong>Analysis Performed:</strong> ${testName}
            </div>
            <div class="result-item">
                <strong>Confidence Level:</strong> ${confidence}%
            </div>
            <div class="result-item">
                <strong>Degrees of Freedom:</strong> ${result.degreesOfFreedom.toFixed(
                  2
                )}
            </div>
            <div class="result-item">
                <strong>Critical T-Value:</strong> ±${result.tCritical.toFixed(
                  4
                )}
            </div>
            <div class="result-item">
                <strong>Margin of Error:</strong> ${result.marginOfError.toFixed(
                  4
                )}
            </div>
            <div class="result-item significant">
                <strong>Confidence Interval:</strong> 
                (${result.lower.toFixed(4)}, ${result.upper.toFixed(4)})
            </div>
            <div class="critical-value">
                <strong>Interpretation:</strong> 
                <p>We are ${confidence}% confident that the true population parameter falls between 
                ${result.lower.toFixed(4)} and ${result.upper.toFixed(4)}.</p>
            </div>
        `;
  }

  // Helper function to get test name
  function getTestName(testType) {
    switch (testType) {
      case TEST_TYPES.ONE_SAMPLE:
        return "One-Sample T-Test";
      case TEST_TYPES.TWO_SAMPLE:
        return "Two-Sample T-Test";
      case TEST_TYPES.PAIRED:
        return "Paired T-Test";
      case TEST_TYPES.CI_SINGLE_MEAN:
        return "Confidence Interval for Single Mean";
      case TEST_TYPES.CI_TWO_MEANS:
        return "Confidence Interval for Difference of Two Means";
      case TEST_TYPES.CI_PAIRED:
        return "Confidence Interval for Paired Differences";
      default:
        return "Statistical Analysis";
    }
  }

  // Helper function to get comparison text
  function getComparisonText(criticalValue, tail) {
    switch (tail) {
      case TAIL_TYPES.LEFT_TAILED:
        return `T < ${criticalValue.toFixed(4)}`;
      case TAIL_TYPES.RIGHT_TAILED:
        return `T > ${criticalValue.toFixed(4)}`;
      case TAIL_TYPES.TWO_TAILED:
        return `|T| > ${criticalValue.toFixed(4)}`;
      default:
        return "";
    }
  }

  // Helper function to get tail description
  function getTailDescription(tail) {
    switch (tail) {
      case TAIL_TYPES.LEFT_TAILED:
        return "left-tailed";
      case TAIL_TYPES.RIGHT_TAILED:
        return "right-tailed";
      case TAIL_TYPES.TWO_TAILED:
        return "two-tailed";
      default:
        return "";
    }
  }

  // Reset the calculator
  function resetCalculator() {
    resultsDiv.classList.add("hidden");
    inputForm.classList.add("hidden");
    testTypeSelect.value = "";
    tTestForm.reset();
  }

  // Initialize the application
  init();
});
