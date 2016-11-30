class NeuronError < RuntimeError; end

class Neuron
  attr_reader :inputs, :output, :weights, :grads
  attr_accessor :alpha

  def initialize(num_inputs, weights: nil)
    @alpha = 0.01
    @weights = weights || ([0.0] + num_inputs.times.map{1})
    @num_inputs = num_inputs.to_i
    validate

    @weights = @weights.map(&:to_f)
    @inputs = @num_inputs.times.map{nil}
    @grads = @num_inputs.times.map{nil}
  end

  def forward(inputs)
    validate_inputs(inputs)
    @inputs = inputs
    @output = sigmoid(sig_in)
  end

  def backward(forward_grad)
    @grads = ([1] + @inputs).map{|ii| forward_grad * ii * ds_dx(sig_in)}
  end

  def backprop(forward_grad = 1)
    backward(forward_grad)
    @weights = @weights.zip(@grads).map{|ww, dw| ww + @alpha * dw}
  end

  private

  def bias
    @weights[0]
  end

  def input_weights
    @weights[1..-1]
  end

  def sigmoid(x)
    1.0 / (1 + Math.exp(-x))
  end

  def ds_dx(x)
    sigmoid(x) * (1 - sigmoid(x))
  end

  def sig_in
    bias + input_weights.zip(@inputs).map{|ww, ii| ii * ww}.reduce(&:+)
  end

  def validate
    @num_inputs > 0 or raise NeuronError, "Num inputs must be positive"
    @weights.length == @num_inputs + 1 or raise NeuronError, "Weights must be length @num_inputs + 1 (first weight is bias weight)"
  end

  def validate_inputs(inputs)
    inputs.is_a?(Array) or raise ArgumentError, "Inputs must be an Array"
    inputs.length == @num_inputs or raise ArgumentError, "Wrong number of inputs (#{inputs.length} for #{@num_inputs})."
  end
end
