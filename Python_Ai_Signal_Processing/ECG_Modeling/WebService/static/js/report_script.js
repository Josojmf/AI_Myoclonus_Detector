document.addEventListener("DOMContentLoaded", function () {
    const pdfUrl = document.getElementById("pdf-url")?.value;
    if (!pdfUrl) {
        console.error("PDF URL not found!");
        return;
    }

    console.log("PDF URL:", pdfUrl);

    const pdfjsLib = window["pdfjs-dist/build/pdf"];
    pdfjsLib.GlobalWorkerOptions.workerSrc = '/static/js/pdf.worker.min.js';

    let pdfDoc = null,
        pageNum = 1,
        pageRendering = false,
        pageNumPending = null,
        scale = 1.5,
        canvas = document.getElementById("pdf-render"),
        ctx = canvas.getContext("2d");

    const container = document.getElementById("pdf-container");

    // Adjust scale dynamically based on container width
    function adjustScale(viewport) {
        const containerWidth = container.offsetWidth - 40; // Padding adjustment
        return containerWidth / viewport.width;
    }

    pdfjsLib
        .getDocument(pdfUrl)
        .promise.then(function (pdfDoc_) {
            console.log("PDF loaded successfully");
            pdfDoc = pdfDoc_;
            document.getElementById("total-pages").textContent = pdfDoc.numPages;
            renderPage(pageNum);
        })
        .catch(function (error) {
            console.error("Error loading PDF:", error);
        });

    function renderPage(num) {
        pageRendering = true;

        pdfDoc
            .getPage(num)
            .then(function (page) {
                const initialViewport = page.getViewport({ scale: 1 });
                const dynamicScale = adjustScale(initialViewport);
                const viewport = page.getViewport({ scale: dynamicScale });

                canvas.height = viewport.height;
                canvas.width = viewport.width;

                const renderContext = {
                    canvasContext: ctx,
                    viewport: viewport,
                };

                const renderTask = page.render(renderContext);

                renderTask.promise.then(function () {
                    pageRendering = false;
                    if (pageNumPending !== null) {
                        renderPage(pageNumPending);
                        pageNumPending = null;
                    }
                });
            })
            .catch(function (error) {
                console.error(`Error rendering page ${num}:`, error);
            });

        document.getElementById("current-page").textContent = num;
    }

    document.getElementById("prev-page").addEventListener("click", function () {
        if (pageNum <= 1 || pageRendering) {
            return;
        }
        pageNum--;
        renderPage(pageNum);
    });

    document.getElementById("next-page").addEventListener("click", function () {
        if (pageNum >= pdfDoc.numPages || pageRendering) {
            return;
        }
        pageNum++;
        renderPage(pageNum);
    });

    // Re-render the page on window resize to maintain scale
    window.addEventListener("resize", function () {
        if (!pageRendering && pdfDoc) {
            renderPage(pageNum);
        }
    });
});
