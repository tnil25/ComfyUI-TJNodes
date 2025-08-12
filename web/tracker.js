// tracker.js

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
	name: "MyTracker.WidgetOrderFix",

	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name === "Tracker") {
			
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function() {
				onNodeCreated?.apply(this, arguments);

				const node = this;
				let existencePollInterval = null;
				let stabilityPollInterval = null;

				// --- 1. Create the button widget FIRST with a placeholder callback ---
                // This ensures it appears at the top of the node.
				const trackButton = node.addWidget("button", "Track Video", null, () => {});

				// --- 2. Create the video player widget SECOND ---
				const previewWidget = node.addDOMWidget("videopreview", "preview", null, { serialize: false });
				previewWidget.element = document.createElement("div");
				previewWidget.element.className = "vhs_preview";
				previewWidget.element.style.width = "100%";
				previewWidget.element.hidden = true;

				const videoEl = document.createElement("video");
				videoEl.controls = true;
				videoEl.loop = true;
				videoEl.muted = true;
				videoEl.style.width = "100%";
				previewWidget.element.appendChild(videoEl);

				previewWidget.computeSize = function(width) {
					if (this.aspectRatio && !this.element.hidden) {
						let height = (node.size[0] - 20) / this.aspectRatio + 10;
						if (!(height > 0)) height = 0;
						return [width, height];
					}
					return [width, -4];
				}
				
				// --- 3. Assign the REAL callback to the button now that all widgets exist ---
				trackButton.callback = async () => {
					// Unload/reset logic
                    if (videoEl.src) {
                        videoEl.pause();
                        videoEl.src = "";
                        previewWidget.element.hidden = true;
                        previewWidget.aspectRatio = null;
                        node.setSize(node.computeSize());
                        app.graph.setDirtyCanvas(true, true);
                    }

					if (existencePollInterval) clearInterval(existencePollInterval);
					if (stabilityPollInterval) clearInterval(stabilityPollInterval);

					try {
						const response = await api.fetchApi(`/tj-nodes/launch-tracker?node_id=${node.id}&t=${Date.now()}`);
						const data = await response.json();
						
						if (data.status !== "success") {
							throw new Error(data.message || "Failed to launch GUI from backend.");
						}

						const filenameToPoll = data.filename;
						console.log(`[Tracker Node] Will poll for file: ${filenameToPoll}`);

						// --- STAGE 1: Poll for file existence ---
						const EXISTENCE_POLL_MS = 2000;
						existencePollInterval = setInterval(async () => {
							try {
								const fileCheckUrl = `/view?filename=${encodeURIComponent(filenameToPoll)}&type=temp&dont_cache=${Date.now()}`;
								const res = await api.fetchApi(fileCheckUrl, { method: 'HEAD' });

								if (res.status === 200) {
									console.log(`[Tracker Node] File found. Now checking for size stability.`);
									clearInterval(existencePollInterval);
									startStabilityPoll(filenameToPoll);
								}
							} catch (error) {
								// 404 is expected, do nothing.
							}
						}, EXISTENCE_POLL_MS);

					} catch (error) {
						console.error("[Tracker Node] Error launching GUI:", error);
					}
				};

				// --- STAGE 2: Poll for file size stability ---
				const startStabilityPoll = (filename) => {
					let lastSize = -1;
					let stableCount = 0;
					const STABILITY_POLL_MS = 500;
					const STABILITY_THRESHOLD = 2;

					stabilityPollInterval = setInterval(async () => {
						try {
							const fileCheckUrl = `/view?filename=${encodeURIComponent(filename)}&type=temp&dont_cache=${Date.now()}`;
							const res = await api.fetchApi(fileCheckUrl, { method: 'HEAD' });

							if (res.status === 200) {
								const currentSize = Number(res.headers.get('Content-Length'));

								if (currentSize > 0 && currentSize === lastSize) {
									stableCount++;
								} else {
									stableCount = 0;
								}
								lastSize = currentSize;

								if (stableCount >= STABILITY_THRESHOLD) {
									console.log(`[Tracker Node] File size is stable at ${currentSize} bytes. Loading video.`);
									clearInterval(stabilityPollInterval);

									// --- Play the video ---
									const videoURL = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=temp`);
									videoEl.src = videoURL;
									previewWidget.element.hidden = false;
									videoEl.play();
									
									videoEl.onloadedmetadata = () => {
										previewWidget.aspectRatio = videoEl.videoWidth / videoEl.videoHeight;
										node.setSize(node.computeSize());
										app.graph.setDirtyCanvas(true, true);
									}
								}
							}
						} catch (error) {
							console.error("[Tracker Node] Error during stability check:", error);
							clearInterval(stabilityPollInterval);
						}
					}, STABILITY_POLL_MS);
				};

				node.onRemoved = () => {
					if (existencePollInterval) clearInterval(existencePollInterval);
					if (stabilityPollInterval) clearInterval(stabilityPollInterval);
				};
			};
		}
	},
});
