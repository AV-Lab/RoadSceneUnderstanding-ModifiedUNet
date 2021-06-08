<h1>Semantic-Segmentation</h1>

<p>Apply Semantic Segmentaion with different approaches using Mapillary Vistas dataset.</p>


<h2>Experiements</h2>
<ul>
  <li>Use <b>U-NET</b> model with some modifications by applying batch normalization technique that increases the training speed in addition to changing the input and output channels. This phase is only testing of the capability of the model to classify only one category (vehicles excluding trucks). The training process contains about 5K~8K images, with validation 1.2~1.5K images. The accuracy per correctly located pixels reach to 87% that almost located all vehicles in the scene. Here is an output of the testing phase that has been applied on a real video in Dubai.</li>
  <img src="https://user-images.githubusercontent.com/20774864/121232074-41417c80-c8a2-11eb-9c91-f891974ea69f.png")
  **Note: Mapillary Vistas dataset contains incorrect annotation and segmentation at some cases.**
</ul>
